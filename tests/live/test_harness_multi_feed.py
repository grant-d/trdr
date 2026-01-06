"""Tests for LiveHarness multi-feed alignment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from trdr.core import Duration, Symbol, Timeframe
from trdr.data import Bar
from trdr.live.config import AlpacaCredentials, LiveConfig
from trdr.live.harness import LiveHarness
from trdr.strategy import BaseStrategy, StrategyConfig
from trdr.strategy.types import DataRequirement, Signal, SignalAction


@dataclass(frozen=True)
class DummyConfig(StrategyConfig):
    """Minimal config for dummy strategy."""


class DummyStrategy(BaseStrategy):
    """Strategy stub that only declares data requirements."""

    def get_data_requirements(self) -> list[DataRequirement]:
        return [
            DataRequirement(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                lookback=self.config.lookback,
                role="primary",
            ),
            DataRequirement(
                symbol=self.config.symbol,
                timeframe=Timeframe.parse("1h"),
                lookback=Duration.parse("6h"),
                role="informative",
            ),
        ]

    def generate_signal(self, bars: dict[str, list[Bar]], position):
        return Signal(action=SignalAction.HOLD, price=0.0, confidence=0.0, reason="test")


def _bar(ts: datetime) -> Bar:
    return Bar(
        timestamp=ts.isoformat(),
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1,
    )


def test_live_harness_aligns_multi_feed_history(monkeypatch):
    symbol = Symbol.parse("crypto:ETH/USD")
    timeframe = Timeframe.parse("15m")
    config = DummyConfig(symbol=symbol, timeframe=timeframe, lookback=Duration.parse("2h"))
    strategy = DummyStrategy(config)

    live_config = LiveConfig(
        mode="paper",
        paper_credentials=AlpacaCredentials(api_key="key", api_secret="secret"),
        symbol=str(symbol),
        timeframe=str(timeframe),
    )
    harness = LiveHarness(live_config, strategy)

    primary_key = harness._primary_requirement.key
    info_key = next(r.key for r in harness._requirements if r.role != "primary")

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    primary_bars = [
        _bar(base),
        _bar(base.replace(minute=15)),
        _bar(base.replace(minute=30)),
        _bar(base.replace(minute=45)),
    ]
    informative_bars = [
        _bar(base),
        _bar(base.replace(hour=1)),
    ]

    async def fake_get_bars_multi(requirements):
        return {
            primary_key: primary_bars,
            info_key: informative_bars,
        }

    monkeypatch.setattr(harness._data_client, "get_bars_multi", fake_get_bars_multi)

    asyncio.run(harness._fetch_bar_history())

    assert set(harness._bars_cache.keys()) == {primary_key, info_key}
    assert len(harness._bars_cache[primary_key]) == len(primary_bars)
    assert len(harness._bars_cache[info_key]) == len(primary_bars)

    aligned = harness._bars_cache[info_key]
    assert all(bar is not None for bar in aligned)
    assert all(bar.timestamp == primary_bars[0].timestamp for bar in aligned)

    # Append a new primary bar at 01:00 and ensure informative aligns forward.
    harness._append_bar(_bar(base.replace(hour=1)))
    assert harness._bars_cache[info_key][-1].timestamp == informative_bars[-1].timestamp

"""Wilder smoothing helpers (RMA/Wilder EMA)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WilderEmaIndicator:
    """Wilder EMA (RMA) for streaming values."""

    period: int
    _seed: list[float] = field(default_factory=list, init=False)
    _value: float | None = field(default=None, init=False)

    def update(self, value: float) -> float:
        if self.period <= 1:
            self._value = float(value)
            return self._value

        if self._value is None:
            self._seed.append(float(value))
            if len(self._seed) < self.period:
                return self._seed[-1]
            self._value = sum(self._seed) / self.period
            return self._value

        self._value = (self._value * (self.period - 1) + float(value)) / self.period
        return self._value

    @property
    def value(self) -> float:
        if self._value is not None:
            return float(self._value)
        return float(self._seed[-1]) if self._seed else 0.0


def wilder_ema_series(values: list[float], period: int) -> list[float]:
    """Compute Wilder EMA series for values."""
    calc = WilderEmaIndicator(period)
    return [calc.update(v) for v in values]

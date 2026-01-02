"""Main orchestration for trading bot."""

import asyncio
import signal
import sys
from pathlib import Path

from .core import BotConfig, load_config
from .data import MarketDataClient, Symbol
from .storage import RunArchive
from .strategy import (
    Position,
    Signal,
    SignalAction,
    VolumeAreaBreakoutConfig,
    VolumeAreaBreakoutStrategy,
)
from .strategy.volume_area_breakout import (
    calculate_volume_profile,
    generate_volume_area_breakout_signal,
)
from .trading import OrderExecutor, OrderStatus
from .ui import (
    LogMessage,
    MarketState,
    MarketUpdate,
    PerformanceState,
    PerformanceUpdate,
    PositionState,
    PositionUpdate,
    StatusUpdate,
    TradingBotApp,
)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: BotConfig):
        """Initialize bot.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.symbol = Symbol.parse(config.strategy.symbol)

        # Components
        self.market = MarketDataClient(config.alpaca, config.cache_dir)
        self.executor = OrderExecutor(config.alpaca)
        self.archive = RunArchive(config.runs_dir)

        # State
        self._running = False
        self._paused = False
        self._position: Position | None = None
        self._pending_order_id: str | None = None
        self._pending_order_side: str | None = None
        self._pending_exit_reason: str | None = None
        self._pending_exit_price: float | None = None
        self._total_pnl = 0.0
        self._daily_pnl = 0.0
        self._trades_today = 0
        self._wins_today = 0

        # TUI
        self._app: TradingBotApp | None = None

    async def run(self) -> None:
        """Run the trading bot."""
        self._running = True

        # Start archive run
        self.archive.start_run(
            symbol=self.symbol.raw,
            config={
                "lookback": self.config.strategy.lookback,
                "atr_threshold": self.config.strategy.atr_threshold,
                "stop_loss_multiplier": self.config.strategy.stop_loss_multiplier,
            },
            mode="paper" if self.config.alpaca.is_paper else "live",
        )

        # Create TUI
        self._app = TradingBotApp()
        self._app.set_callbacks(
            on_pause=self._on_pause,
            on_resume=self._on_resume,
            on_stop=self._on_stop,
        )

        # Run with TaskGroup
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._run_tui())
                tg.create_task(self._price_loop())
                tg.create_task(self._trading_loop())
                tg.create_task(self._order_loop())
        except* Exception as eg:
            for exc in eg.exceptions:
                self._log(f"Error: {exc}", "error")
        finally:
            self._cleanup()

    async def _run_tui(self) -> None:
        """Run the TUI application."""
        if self._app:
            await self._app.run_async()
            # TUI exited - stop everything
            self._running = False

    async def _price_loop(self) -> None:
        """Fetch price updates every 1s."""
        interval = self.config.loops.price_interval

        while self._running:
            try:
                if not self._paused:
                    quote = await self.market.get_current_price(self.symbol)
                    bars = await self.market.get_bars(
                        self.symbol,
                        lookback=self.config.strategy.lookback,
                    )

                    if bars:
                        profile = calculate_volume_profile(bars)
                        prev_close = bars[-2].close if len(bars) > 1 else quote.price

                        if prev_close:
                            change_pct = ((quote.price - prev_close) / prev_close) * 100
                        else:
                            change_pct = 0

                        self._post_message(
                            MarketUpdate(
                                MarketState(
                                    symbol=self.symbol.raw,
                                    price=quote.price,
                                    change_pct=change_pct,
                                    poc=profile.poc,
                                    vah=profile.vah,
                                    val=profile.val,
                                    volume=bars[-1].volume if bars else 0,
                                )
                            )
                        )

            except Exception as e:
                self._log(f"Price fetch error: {e}", "error")

            await asyncio.sleep(interval)

    async def _trading_loop(self) -> None:
        """Generate signals and execute trades every 5s."""
        interval = self.config.loops.trading_interval

        # Wait for initial data
        await asyncio.sleep(2)
        self._post_message(StatusUpdate("RUNNING"))
        self._log("Bot started, monitoring for signals", "success")

        while self._running:
            try:
                if not self._paused and not self._pending_order_id:
                    await self._check_signals()

            except Exception as e:
                self._log(f"Trading error: {e}", "error")

            await asyncio.sleep(interval)

    async def _order_loop(self) -> None:
        """Monitor pending orders every 0.5s."""
        interval = self.config.loops.order_interval

        while self._running:
            try:
                if self._pending_order_id:
                    await self._check_order()

                # Update position display
                await self._update_position_display()

            except Exception as e:
                self._log(f"Order check error: {e}", "error")

            await asyncio.sleep(interval)

    async def _check_signals(self) -> None:
        """Check for trading signals and execute."""
        bars = await self.market.get_bars(
            self.symbol,
            lookback=self.config.strategy.lookback,
        )

        if len(bars) < 20:
            return

        signal = generate_volume_area_breakout_signal(
            bars=bars,
            position=self._position,
            atr_threshold=self.config.strategy.atr_threshold,
            stop_loss_multiplier=self.config.strategy.stop_loss_multiplier,
        )

        if signal.action == SignalAction.BUY and not self._position:
            await self._enter_position(signal)

        elif signal.action == SignalAction.CLOSE and self._position:
            await self._exit_position(signal)

    async def _enter_position(self, signal) -> None:
        """Enter a new position."""
        try:
            # Calculate position size (use fixed size for now)
            qty = self.config.strategy.position_size

            order = await self.executor.submit_market_order(
                symbol=self.symbol.raw,
                qty=qty,
                side="buy",
            )

            self._pending_order_id = order.id
            self._pending_order_side = "buy"
            self._log(f"BUY order submitted: {qty} @ ~${signal.price:.2f}", "info")

            # Store planned position
            self._position = Position(
                symbol=self.symbol.raw,
                side="long",
                size=qty,
                entry_price=signal.price,
                stop_loss=signal.stop_loss or 0,
                take_profit=signal.take_profit,
            )

            # Archive trade
            profile = calculate_volume_profile(
                await self.market.get_bars(self.symbol, lookback=self.config.strategy.lookback)
            )
            self.archive.record_trade(
                action="buy",
                symbol=self.symbol.raw,
                price=signal.price,
                qty=qty,
                reason=signal.reason,
                poc=profile.poc,
                va_high=profile.vah,
                va_low=profile.val,
            )

        except Exception as e:
            self._log(f"Order failed: {e}", "error")
            self._position = None

    async def _exit_position(self, signal) -> None:
        """Exit current position."""
        if not self._position:
            return

        try:
            order = await self.executor.close_position(self.symbol.raw)
            if order:
                self._pending_order_id = order.id
                self._pending_order_side = "sell"
                self._pending_exit_reason = signal.reason
                self._pending_exit_price = signal.price
                self._log(f"SELL order submitted @ ~${signal.price:.2f}", "info")

        except Exception as e:
            self._log(f"Close failed: {e}", "error")

    async def _check_order(self) -> None:
        """Check status of pending order."""
        if not self._pending_order_id:
            return

        try:
            order = await self.executor.get_order(self._pending_order_id)

            if order.status == OrderStatus.FILLED:
                filled_price = order.filled_price
                self._log(
                    f"Order filled: {order.filled_qty} @ ${filled_price:.2f}"
                    if filled_price is not None
                    else f"Order filled: {order.filled_qty}",
                    "success",
                )

                # Update position with actual fill price
                if self._position and order.side == "buy":
                    if filled_price is not None:
                        self._position.entry_price = filled_price

                # P&L and position clearing deferred until sell is confirmed filled.
                # This prevents booking incorrect P&L if the close order is rejected.
                if self._position and order.side == "sell":
                    exit_price = filled_price or self._pending_exit_price or self._position.entry_price
                    pnl = (exit_price - self._position.entry_price) * self._position.size

                    self.archive.record_trade(
                        action="sell",
                        symbol=self.symbol.raw,
                        price=exit_price,
                        qty=self._position.size,
                        reason=self._pending_exit_reason or "order_filled",
                        pnl=pnl,
                    )

                    self._total_pnl += pnl
                    self._daily_pnl += pnl
                    self._trades_today += 1
                    if pnl > 0:
                        self._wins_today += 1

                    self._position = None
                    self._update_performance_display()

                self._pending_order_id = None
                self._pending_order_side = None
                self._pending_exit_reason = None
                self._pending_exit_price = None

            elif order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                self._log(f"Order {order.status.value}", "warning")
                self._pending_order_id = None
                self._pending_order_side = None
                self._pending_exit_reason = None
                self._pending_exit_price = None
                # Only clear position for failed buys. For failed sells, keep position
                # intact so we can retry the exit - position is still live on exchange.
                if order.side == "buy":
                    self._position = None

        except Exception as e:
            self._log(f"Order check failed: {e}", "error")

    async def _update_position_display(self) -> None:
        """Update position panel in TUI."""
        if not self._position:
            self._post_message(
                PositionUpdate(
                    PositionState(
                        side="NONE",
                        size=0,
                        entry_price=0,
                        current_price=0,
                        unrealized_pnl=0,
                        stop_loss=None,
                        take_profit=None,
                    )
                )
            )
            return

        try:
            quote = await self.market.get_current_price(self.symbol)
            unrealized = (quote.price - self._position.entry_price) * self._position.size

            self._post_message(
                PositionUpdate(
                    PositionState(
                        side="LONG" if self._position.side == "long" else "SHORT",
                        size=self._position.size,
                        entry_price=self._position.entry_price,
                        current_price=quote.price,
                        unrealized_pnl=unrealized,
                        stop_loss=self._position.stop_loss,
                        take_profit=self._position.take_profit,
                    )
                )
            )
        except Exception:
            pass

    def _update_performance_display(self) -> None:
        """Update performance panel in TUI."""
        win_rate = self._wins_today / self._trades_today if self._trades_today else 0

        self._post_message(
            PerformanceUpdate(
                PerformanceState(
                    total_pnl=self._total_pnl,
                    daily_pnl=self._daily_pnl,
                    win_rate=win_rate,
                    total_trades=self._trades_today,
                    winning_trades=self._wins_today,
                )
            )
        )

    def _post_message(self, message) -> None:
        """Post message to TUI."""
        if self._app:
            self._app.post_message(message)

    def _log(self, text: str, level: str = "info") -> None:
        """Log message to TUI."""
        self._post_message(LogMessage(text, level))

    def _on_pause(self) -> None:
        """Handle pause from TUI."""
        self._paused = True

    def _on_resume(self) -> None:
        """Handle resume from TUI."""
        self._paused = False

    def _on_stop(self) -> None:
        """Handle stop from TUI."""
        self._running = False

    def _cleanup(self) -> None:
        """Clean up on exit."""
        metrics = self.archive.end_run()
        if metrics:
            print("\nRun complete:")
            print(f"  Trades: {metrics.total_trades}")
            print(f"  Win Rate: {metrics.win_rate:.1%}")
            print(f"  P&L: ${metrics.total_pnl:.2f}")


def main() -> None:
    """Entry point."""
    try:
        config = load_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Ensure .env file exists with ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)

    bot = TradingBot(config)

    # Handle SIGINT gracefully
    def handle_sigint(sig, frame):
        bot._running = False

    signal.signal(signal.SIGINT, handle_sigint)

    asyncio.run(bot.run())


if __name__ == "__main__":
    main()

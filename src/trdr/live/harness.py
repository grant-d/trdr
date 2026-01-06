"""Live trading harness - poll-based orchestrator."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..core import Symbol
from ..core.config import AlpacaConfig
from ..data import AlpacaDataClient, Bar
from .config import LiveConfig
from .exchange.alpaca import AlpacaExchange
from .exchange.base import ExchangeError
from .exchange.types import (
    HydraBar,
    HydraFill,
    HydraOrderRequest,
    HydraOrderSide,
    HydraOrderType,
)
from .orders.manager import OrderManager
from .orders.retry import RetryPolicy
from .orders.types import LiveOrder
from .safety.circuit_breaker import BreakerTrip, CircuitBreaker
from .state.context import LiveContextBuilder, LiveRuntimeContext
from .state.reconciler import StateReconciler

if TYPE_CHECKING:
    from ..strategy import BaseStrategy
    from ..strategy.types import Signal

logger = logging.getLogger(__name__)
project_root = Path(__file__).parent.parent.parent.parent


@dataclass
class HarnessState:
    """Internal state of the live harness.

    Args:
        running: Whether harness is running
        paused: Whether trading is paused
        last_bar_time: Timestamp of last processed bar
        last_signal: Last signal from strategy
        bars_processed: Total bars processed
        start_time: When harness started
        errors: Recent errors
    """

    running: bool = False
    paused: bool = False
    last_bar_time: str = ""
    last_signal: "Signal | None" = None
    bars_processed: int = 0
    start_time: datetime | None = None
    errors: list[str] = field(default_factory=list)


class LiveHarness:
    """Poll-based live trading orchestrator.

    Coordinates exchange, strategy, orders, and safety systems.

    Example:
        config = LiveConfig.from_env()
        harness = LiveHarness(config, strategy)
        await harness.start()  # Runs until stopped
    """

    def __init__(
        self,
        config: LiveConfig,
        strategy: "BaseStrategy",
        on_signal: Callable[["Signal"], None] | None = None,
        on_fill: Callable[[LiveOrder, HydraFill], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        """Initialize live harness.

        Args:
            config: Live trading configuration
            strategy: Trading strategy
            on_signal: Optional callback for signals
            on_fill: Optional callback for fills
            on_error: Optional callback for errors
        """
        self._config = config
        self._strategy = strategy
        self._on_signal = on_signal
        self._on_fill = on_fill
        self._on_error = on_error

        # Validate config
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid config: {errors}")

        # Initialize components
        self._exchange = AlpacaExchange(
            credentials=config.credentials,
            paper=config.is_paper,
        )
        self._order_manager = OrderManager(
            exchange=self._exchange,
            retry_policy=RetryPolicy(max_attempts=config.max_retries),
            audit_log_path=config.log_file,
        )
        self._circuit_breaker = CircuitBreaker(
            limits=config.risk_limits,
            on_trip=self._handle_breaker_trip,
        )
        self._reconciler = StateReconciler(
            exchange=self._exchange,
            order_manager=self._order_manager,
        )
        self._context_builder = LiveContextBuilder(
            exchange=self._exchange,
            order_manager=self._order_manager,
            circuit_breaker=self._circuit_breaker,
            symbol=config.symbol,
            strategy_name=strategy.name,
        )

        # State
        self._state = HarnessState()
        self._stop_event = asyncio.Event()

        # Data client for bar history (shared with backtester, uses caching)
        data_config = AlpacaConfig(
            api_key=config.credentials.api_key,
            secret_key=config.credentials.api_secret,
        )
        cache_dir = project_root / "data/cache"
        self._data_client = AlpacaDataClient(data_config, cache_dir)

        # Data requirements and bar cache per feed
        self._requirements = strategy.get_data_requirements()
        self._primary_requirement = next(r for r in self._requirements if r.role == "primary")
        self._bars_cache: dict[str, list[Bar | None]] = {}
        self._bars_raw: dict[str, list[Bar]] = {}
        self._informative_indices: dict[str, int] = {}
        self._bars_cache_max: dict[str, int] = {
            r.key: r.lookback_bars for r in self._requirements
        }

        # Position tracking for trade PnL calculation
        self._position_entries: dict[str, float] = {}  # symbol -> avg entry price
        self._position_stops: dict[str, float] = {}  # symbol -> stop loss price
        self._position_targets: dict[str, float | None] = {}  # symbol -> take profit

    @property
    def state(self) -> HarnessState:
        """Get current harness state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if harness is running."""
        return self._state.running

    @property
    def is_paused(self) -> bool:
        """Check if trading is paused."""
        return self._state.paused

    async def start(self) -> None:
        """Start the live trading loop.

        Runs until stop() is called or circuit breaker trips.
        """
        logger.info(f"Starting live harness for {self._config.symbol}")
        logger.info(f"Mode: {'PAPER' if self._config.is_paper else 'LIVE'}")

        try:
            # Connect to exchange
            await self._exchange.connect()
            logger.info("Connected to exchange")

            # Reconcile state
            result = await self._reconciler.reconcile(self._config.symbol)
            if not result.success:
                raise RuntimeError(f"Reconciliation failed: {result.errors}")

            # Initialize circuit breaker
            if result.account:
                self._circuit_breaker.initialize(result.account.equity)

            # Register fill callback
            self._order_manager.on_fill(self._handle_fill)

            # Fetch initial bar history
            await self._fetch_bar_history()

            # Start main loop
            self._state.running = True
            self._state.start_time = datetime.now(UTC)
            self._stop_event.clear()

            await self._run_loop()

        except Exception as e:
            logger.exception(f"Harness error: {e}")
            self._state.errors.append(str(e))
            if self._on_error:
                self._on_error(e)
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the live trading loop gracefully."""
        logger.info("Stopping harness...")
        self._stop_event.set()

    def pause(self) -> None:
        """Pause trading (still monitors but no new orders)."""
        self._state.paused = True
        logger.info("Trading paused")

    def resume(self) -> None:
        """Resume trading after pause."""
        self._state.paused = False
        logger.info("Trading resumed")

    async def _run_loop(self) -> None:
        """Main trading loop."""
        logger.info("Entering main loop")

        while not self._stop_event.is_set():
            try:
                # Check circuit breaker
                can_trade, reason = self._circuit_breaker.check_can_trade()
                if not can_trade:
                    logger.warning(f"Trading blocked: {reason}")
                    await asyncio.sleep(self._config.poll_interval_seconds)
                    continue

                # Update feeds and check for new primary bar
                primary_bar = await self._update_feeds()

                if not primary_bar:
                    # No new primary bar
                    await asyncio.sleep(self._config.poll_interval_seconds)
                    continue

                self._state.last_bar_time = primary_bar.timestamp
                self._state.bars_processed += 1

                # Refresh order states
                await self._order_manager.refresh_orders()

                # Build context and run strategy
                await self._process_bar(primary_bar)

            except ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                self._state.errors.append(str(e))
                await asyncio.sleep(self._config.poll_interval_seconds)

            except Exception as e:
                logger.exception(f"Loop error: {e}")
                self._state.errors.append(str(e))
                if self._on_error:
                    self._on_error(e)
                await asyncio.sleep(self._config.poll_interval_seconds)

            # Wait for next poll
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._config.poll_interval_seconds,
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Continue loop

    async def _update_feeds(self) -> HydraBar | None:
        """Fetch latest bars for all requirements and update raw cache.

        Returns:
            New primary bar if detected, else None.
        """
        new_primary_bar = None
        results = {}

        # Fetch latest bar for each requirement
        # Note: sequentially for now, could be parallelized
        for req in self._requirements:
            try:
                symbol = Symbol.parse(req.symbol) if isinstance(req.symbol, str) else req.symbol
                bars = await self._data_client.get_bars(
                    symbol=symbol,
                    lookback=1,
                    timeframe=req.timeframe,
                    # Live loop must bypass cache freshness or 15m bars can stall for 1h.
                    force_refresh=True,
                )
                if bars:
                    results[req.key] = bars[-1]
            except Exception as e:
                logger.warning(f"Failed to fetch bar for {req.key}: {e}")

        # Process results
        for req in self._requirements:
            bar = results.get(req.key)
            if not bar:
                continue

            # Check if primary is new
            if req.role == "primary":
                if bar.timestamp != self._state.last_bar_time:
                    new_primary_bar = HydraBar(
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=float(bar.volume),
                    )
                continue

            # Update secondary feeds if new
            current_raw = self._bars_raw.get(req.key, [])
            if not current_raw or current_raw[-1].timestamp != bar.timestamp:
                # Convert to internal Bar
                trdr_bar = Bar(
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=int(bar.volume),
                )

                if not current_raw:
                    current_raw = []

                current_raw.append(trdr_bar)
                max_len = self._bars_cache_max.get(req.key, 100)
                if len(current_raw) > max_len:
                    current_raw = current_raw[-max_len:]
                self._bars_raw[req.key] = current_raw

        return new_primary_bar

    async def _fetch_bar_history(self) -> None:
        """Fetch historical bars using shared data client with caching."""
        logger.info(f"Fetching bar history for {self._config.symbol}...")

        from ..backtest import align_feeds

        bars = await self._data_client.get_bars_multi(self._requirements)
        primary_key = self._primary_requirement.key
        primary_bars = bars.get(primary_key, [])

        # Store raw bars and build aligned caches
        self._bars_raw = bars
        self._bars_cache = {primary_key: list(primary_bars)}
        self._informative_indices = {}

        for req in self._requirements:
            if req.role == "primary":
                continue
            raw_bars = bars.get(req.key, [])
            self._bars_cache[req.key] = align_feeds(primary_bars, raw_bars)
            last_idx = -1
            if raw_bars and primary_bars:
                for i in range(len(raw_bars) - 1, -1, -1):
                    if raw_bars[i].timestamp <= primary_bars[-1].timestamp:
                        last_idx = i
                        break
            self._informative_indices[req.key] = last_idx

        logger.info(f"Loaded {len(primary_bars)} historical bars")

    def _append_bar(self, bar: Bar) -> None:
        """Append a bar to the primary cache and align informative feeds."""
        primary_key = self._primary_requirement.key
        primary_cache = self._bars_cache.get(primary_key, [])
        primary_raw = self._bars_raw.get(primary_key, [])

        if primary_cache and primary_cache[-1] and primary_cache[-1].timestamp == bar.timestamp:
            primary_cache[-1] = bar
            primary_raw[-1] = bar
        else:
            primary_cache.append(bar)
            primary_raw.append(bar)

        self._bars_cache[primary_key] = primary_cache
        self._bars_raw[primary_key] = primary_raw

        # Align informative feeds to the new primary bar
        for req in self._requirements:
            if req.role == "primary":
                continue
            key = req.key
            raw_bars = self._bars_raw.get(key, [])
            idx = self._informative_indices.get(key, -1)

            while idx + 1 < len(raw_bars) and raw_bars[idx + 1].timestamp <= bar.timestamp:
                idx += 1

            aligned_bar = raw_bars[idx] if idx >= 0 and raw_bars[idx].timestamp <= bar.timestamp else None
            aligned_cache = self._bars_cache.get(key, [])
            aligned_cache.append(aligned_bar)

            self._informative_indices[key] = idx
            self._bars_cache[key] = aligned_cache

        # Trim caches to configured lookbacks
        for key, max_bars in self._bars_cache_max.items():
            cache = self._bars_cache.get(key, [])
            if len(cache) > max_bars:
                self._bars_cache[key] = cache[-max_bars:]

        primary_max = self._bars_cache_max.get(primary_key)
        if primary_max and len(primary_raw) > primary_max:
            self._bars_raw[primary_key] = primary_raw[-primary_max:]

    async def _process_bar(self, bar: HydraBar) -> None:
        """Process a new bar through the strategy.

        Args:
            bar: New bar to process
        """
        from ..strategy.types import Position, SignalAction

        # Build context
        context = await self._context_builder.build(bar)

        # Set context on strategy
        self._strategy.context = context

        # Get current position with tracked stops
        symbol = self._config.symbol
        position = context.positions.get(symbol)
        strategy_position = None
        if position:
            strategy_position = Position(
                symbol=position.symbol,
                side=position.side,
                size=position.quantity,
                entry_price=position.avg_entry_price,
                stop_loss=self._position_stops.get(symbol, 0.0),
                take_profit=self._position_targets.get(symbol),
            )

        # Convert HydraBar to Bar format and append to cache
        trdr_bar = Bar(
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=int(bar.volume),
        )
        self._append_bar(trdr_bar)

        # Build bars dict for strategy using full history
        bars_dict = {key: list(bars) for key, bars in self._bars_cache.items()}

        # Generate signal
        signal = self._strategy.generate_signal(bars_dict, strategy_position)
        self._state.last_signal = signal

        if self._on_signal:
            self._on_signal(signal)

        # Skip if paused or HOLD
        if self._state.paused:
            return

        if signal.action == SignalAction.HOLD:
            return

        # Process signal into orders
        await self._execute_signal(signal, context, bar)

    async def _execute_signal(
        self,
        signal: "Signal",
        context: LiveRuntimeContext,
        bar: HydraBar,
    ) -> None:
        """Execute a trading signal.

        Args:
            signal: Signal to execute
            context: Current context
            bar: Current bar
        """
        from ..strategy.types import SignalAction

        symbol = self._config.symbol
        position = context.positions.get(symbol)

        if signal.action == SignalAction.CLOSE:
            # Close existing position
            if position and position.quantity > 0:
                side = HydraOrderSide.SELL if position.side == "long" else HydraOrderSide.BUY
                order = HydraOrderRequest(
                    symbol=symbol,
                    side=side,
                    qty=position.quantity,
                    order_type=HydraOrderType.MARKET,
                )
                await self._order_manager.submit_order(order, reason=signal.reason)
            return

        if signal.action in (SignalAction.BUY, SignalAction.SELL):
            # Calculate quantity
            qty = self._calculate_quantity(signal, context, bar.close)
            if qty <= 0:
                logger.warning("Calculated quantity <= 0, skipping")
                return

            # Check position size limits
            position_value = qty * bar.close
            allowed, reason = self._circuit_breaker.check_position_size(
                position_value, context.equity
            )
            if not allowed:
                logger.warning(f"Position size rejected: {reason}")
                return

            # Close opposite position first if needed
            if position:
                is_opposite = (signal.action == SignalAction.BUY and position.side == "short") or (
                    signal.action == SignalAction.SELL and position.side == "long"
                )
                if is_opposite:
                    close_side = (
                        HydraOrderSide.SELL if position.side == "long" else HydraOrderSide.BUY
                    )
                    close_order = HydraOrderRequest(
                        symbol=symbol,
                        side=close_side,
                        qty=position.quantity,
                        order_type=HydraOrderType.MARKET,
                    )
                    await self._order_manager.submit_order(close_order, reason="reverse_position")

            # Build entry order
            side = HydraOrderSide.BUY if signal.action == SignalAction.BUY else HydraOrderSide.SELL

            if signal.limit_price:
                order_type = HydraOrderType.LIMIT
                if signal.stop_price:
                    order_type = HydraOrderType.STOP_LIMIT
            elif signal.stop_price:
                order_type = HydraOrderType.STOP
            else:
                order_type = HydraOrderType.MARKET

            order = HydraOrderRequest(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=order_type,
                limit_price=signal.limit_price,
                stop_price=signal.stop_price,
            )

            # Track position info for later trade recording
            self._position_stops[symbol] = signal.stop_loss if signal.stop_loss else 0.0
            self._position_targets[symbol] = signal.take_profit

            await self._order_manager.submit_order(order, reason=signal.reason)

    def _calculate_quantity(
        self,
        signal: "Signal",
        context: LiveRuntimeContext,
        price: float,
    ) -> float:
        """Calculate order quantity from signal.

        Args:
            signal: Trading signal
            context: Current context
            price: Current price

        Returns:
            Quantity to order
        """
        if signal.quantity:
            return signal.quantity

        # Calculate from position size percentage
        equity = context.equity
        max_position = equity * (self._config.risk_limits.max_position_pct / 100)
        target_value = max_position * signal.position_size_pct
        qty = target_value / price

        return qty

    def _handle_fill(self, order: LiveOrder, fill: HydraFill) -> None:
        """Handle order fill event.

        Args:
            order: Filled order
            fill: Fill details
        """
        logger.info(f"Fill: {fill.side.value} {fill.qty} {fill.symbol} @ ${fill.price:.2f}")

        if self._on_fill:
            self._on_fill(order, fill)

        symbol = fill.symbol

        # Track entry or record exit
        if order.reason in ("stop_loss", "take_profit", "close", "reverse_position"):
            # This is an exit, record the trade
            entry_price = self._position_entries.get(symbol, fill.price)
            is_long = fill.side == HydraOrderSide.SELL  # Selling means was long

            if is_long:
                pnl = (fill.price - entry_price) * fill.qty
            else:
                pnl = (entry_price - fill.price) * fill.qty

            logger.info(
                f"Trade closed: entry=${entry_price:.2f}, exit=${fill.price:.2f}, "
                f"PnL=${pnl:.2f}"
            )

            # Record trade in context builder (which also updates circuit breaker)
            from .state.context import LiveTrade

            trade = LiveTrade(
                symbol=symbol,
                side="long" if is_long else "short",
                quantity=fill.qty,
                entry_price=entry_price,
                exit_price=fill.price,
                pnl=pnl,
                entry_time=datetime.now(UTC),  # Approximate, could track actual
                exit_time=datetime.now(UTC),
                reason=order.reason or "",
            )
            self._context_builder.record_trade(trade)

            # Clear position tracking
            self._position_entries.pop(symbol, None)
            self._position_stops.pop(symbol, None)
            self._position_targets.pop(symbol, None)
        else:
            # This is an entry, track it
            self._position_entries[symbol] = fill.price
            logger.debug(f"Position entry tracked: {symbol} @ ${fill.price:.2f}")

    def _handle_breaker_trip(self, trip: BreakerTrip) -> None:
        """Handle circuit breaker trip.

        Args:
            trip: Trip details
        """
        logger.critical(
            f"CIRCUIT BREAKER TRIPPED: {trip.reason} "
            f"(value={trip.value:.2f}, limit={trip.limit:.2f})"
        )
        self._state.paused = True

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._state.running = False

        try:
            # Cancel pending orders
            cancelled = await self._order_manager.cancel_all(self._config.symbol)
            logger.info(f"Cancelled {cancelled} pending orders")
        except Exception as e:
            logger.warning(f"Error cancelling orders: {e}")

        try:
            await self._exchange.disconnect()
            logger.info("Disconnected from exchange")
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

    def get_status(self) -> dict:
        """Get current harness status.

        Returns:
            Dict with status information
        """
        return {
            "running": self._state.running,
            "paused": self._state.paused,
            "mode": "paper" if self._config.is_paper else "live",
            "symbol": self._config.symbol,
            "bars_processed": self._state.bars_processed,
            "last_bar_time": self._state.last_bar_time,
            "start_time": (self._state.start_time.isoformat() if self._state.start_time else None),
            "circuit_breaker": self._circuit_breaker.get_status(),
            "active_orders": len(self._order_manager.active_orders),
            "recent_errors": self._state.errors[-5:],
        }

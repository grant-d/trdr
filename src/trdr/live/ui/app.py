"""Live trading dashboard application."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Footer

from .widgets import (
    CircuitPanel,
    LogPanel,
    OrdersPanel,
    PositionPanel,
    SignalsPanel,
    StatusHeader,
)

if TYPE_CHECKING:
    from ..harness import LiveHarness

CSS_PATH = Path(__file__).parent / "styles.tcss"


class LiveDashboard(App):
    """Live trading TUI dashboard."""

    CSS_PATH = CSS_PATH

    BINDINGS = [
        ("p", "pause", "Pause"),
        ("r", "resume", "Resume"),
        ("s", "stop", "Stop"),
        ("b", "reset_breaker", "Reset Breaker"),
        ("x", "screenshot", "Screenshot"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, harness: LiveHarness):
        """Initialize dashboard.

        Args:
            harness: Live harness instance
        """
        super().__init__()
        self._harness = harness
        self._refresh_task: asyncio.Task | None = None
        self._ready_event: asyncio.Event | None = None
        self._harness_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        """Create the layout."""
        yield StatusHeader(id="header")
        with Container(id="main"):
            with Horizontal(id="top-row"):
                yield PositionPanel(id="position", classes="panel")
                yield CircuitPanel(id="circuit", classes="panel")
            with Horizontal(id="mid-row"):
                yield OrdersPanel(id="orders", classes="panel")
                yield SignalsPanel(id="signals", classes="panel")
            yield LogPanel(id="log", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.title = "TRDR Live Trading"

        # Set initial state
        config = self._harness._config
        header = self.query_one("#header", StatusHeader)
        header.symbol = config.symbol
        header.timeframe = str(self._harness._primary_requirement.timeframe)
        header.mode = "PAPER" if config.is_paper else "LIVE"
        header.state = "INIT"

        # Set circuit limits
        circuit = self.query_one("#circuit", CircuitPanel)
        circuit.max_drawdown_pct = config.risk_limits.max_drawdown_pct
        circuit.max_daily_loss = 0.0

        # Log startup
        log = self.query_one("#log", LogPanel)
        log.add_log("Dashboard started", "info")
        log.add_log(f"Symbol: {config.symbol}", "info")
        log.add_log(f"Mode: {'PAPER' if config.is_paper else 'LIVE'}", "info")
        log.add_log("Waiting for harness to connect...", "info")

        # Register harness callbacks
        self._harness._on_signal = self._on_signal
        self._harness._on_fill = self._on_fill
        self._harness._on_error = self._on_error

        # Start refresh loop
        self._refresh_task = asyncio.create_task(self._refresh_loop())

        # Signal ready (create event here where loop exists)
        self._ready_event = asyncio.Event()
        self._ready_event.set()

    async def on_unmount(self) -> None:
        """Called when app is unmounted."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

    async def wait_until_ready(self) -> None:
        """Wait for UI to be mounted and callbacks ready."""
        while self._ready_event is None:
            await asyncio.sleep(0.05)
        await self._ready_event.wait()

    async def _refresh_loop(self) -> None:
        """Periodically refresh status from harness."""
        while True:
            try:
                await asyncio.sleep(1.0)
                self._refresh_status()
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Ignore refresh errors

    def _refresh_status(self) -> None:
        """Refresh all panels from harness status."""
        try:
            snapshot = self._harness.get_ui_snapshot()
        except Exception:
            return  # Harness not ready yet

        # Update header
        header = self.query_one("#header", StatusHeader)
        old_state = header.state
        status = snapshot["status"]
        if status["running"]:
            header.state = "PAUSED" if status["paused"] else "RUNNING"
        elif old_state in ("RUNNING", "PAUSED"):
            # Only show STOPPED if we were actually running
            # Keep INIT while harness is starting, ERROR if crashed
            header.state = "STOPPED"

        # Log state changes
        if old_state != header.state:
            log = self.query_one("#log", LogPanel)
            if header.state == "RUNNING" and old_state == "INIT":
                log.add_log("Harness connected", "success")
            log.add_log(f"State: {header.state}", "info")

        cb_status = status.get("circuit_breaker", {})
        header.circuit_state = cb_status.get("state", "closed")
        header.bars_processed = status.get("bars_processed", 0)

        header.price = snapshot.get("price") or 0.0

        # Update circuit panel
        circuit = self.query_one("#circuit", CircuitPanel)
        circuit.state = cb_status.get("state", "closed")
        circuit.equity = cb_status.get("equity", 0)
        circuit.high_water_mark = cb_status.get("high_water_mark", 0)
        circuit.drawdown_pct = cb_status.get("drawdown_pct", 0)
        circuit.daily_pnl = cb_status.get("daily_pnl", 0)
        limits = cb_status.get("limits", {})
        equity = cb_status.get("equity", 0)
        max_daily_loss_pct = limits.get("max_daily_loss_pct", 0)
        circuit.max_daily_loss = (equity * max_daily_loss_pct / 100) if equity else 0.0

        # Update orders panel
        orders_panel = self.query_one("#orders", OrdersPanel)
        orders_panel.update_orders(snapshot.get("orders", []))

        # Update position panel from context if available
        self._update_position_panel(snapshot)

    def _update_position_panel(self, snapshot: dict) -> None:
        """Update position panel from harness state."""
        try:
            position_panel = self.query_one("#position", PositionPanel)

            symbol = self._harness._config.symbol
            position = snapshot.get("position")
            if position:
                position_panel.side = position.side.upper()
                position_panel.qty = position.quantity
                position_panel.entry = position.avg_entry_price
                position_panel.stop = self._harness._position_stops.get(symbol, 0)
                position_panel.target = self._harness._position_targets.get(symbol) or 0
                position_panel.current = snapshot.get("price") or 0.0
            else:
                position_panel.side = "FLAT"
                position_panel.qty = 0
                position_panel.entry = 0
                position_panel.current = 0
        except Exception:
            pass  # Widget not ready

    def _on_signal(self, signal) -> None:
        """Handle signal from harness."""
        from ...strategy.types import SignalAction

        action_map = {
            SignalAction.BUY: "LONG",
            SignalAction.SELL: "SHORT",
            SignalAction.CLOSE: "CLOSE",
            SignalAction.HOLD: "HOLD",
        }
        action = action_map.get(signal.action, str(signal.action))

        signals_panel = self.query_one("#signals", SignalsPanel)
        signals_panel.add_signal(action, signal.confidence or 0, signal.reason or "")

        if signal.action != SignalAction.HOLD:
            log = self.query_one("#log", LogPanel)
            log.add_log(f"Signal: {action} confidence={signal.confidence:.2f}", "info")

    def _on_fill(self, order, fill) -> None:
        """Handle fill from harness."""
        log = self.query_one("#log", LogPanel)
        side = fill.side.value if hasattr(fill.side, "value") else str(fill.side)
        log.add_log(
            f"Fill: {side} {fill.qty:.4f} {fill.symbol} @ ${fill.price:,.2f}",
            "success",
        )

    def _on_error(self, error: Exception) -> None:
        """Handle error from harness."""
        log = self.query_one("#log", LogPanel)
        log.add_log(f"Error: {error}", "error")

    def action_pause(self) -> None:
        """Pause trading."""
        self._harness.pause()
        self.query_one("#header", StatusHeader).state = "PAUSED"
        self.query_one("#log", LogPanel).add_log("Trading paused", "warning")

    def action_resume(self) -> None:
        """Resume trading."""
        self._harness.resume()
        self.query_one("#header", StatusHeader).state = "RUNNING"
        self.query_one("#log", LogPanel).add_log("Trading resumed", "success")

    async def action_stop(self) -> None:
        """Stop trading and exit."""
        self.query_one("#header", StatusHeader).state = "STOPPED"
        self.query_one("#log", LogPanel).add_log("Stopping harness...", "warning")
        await self._harness.stop()
        self.exit()

    async def action_quit(self) -> None:
        """Quit application."""
        await self.action_stop()

    def action_reset_breaker(self) -> None:
        """Reset circuit breaker if open."""
        cb = self._harness._circuit_breaker
        if cb._state.value == "open":
            cb.reset()
            self.query_one("#log", LogPanel).add_log("Circuit breaker reset", "warning")
            self.query_one("#header", StatusHeader).circuit_state = "closed"
            self.query_one("#circuit", CircuitPanel).state = "closed"
        else:
            self.query_one("#log", LogPanel).add_log("Breaker not open", "info")

    def action_screenshot(self) -> None:
        """Save screenshot of current dashboard."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trdr_screenshot_{timestamp}.svg"
        self.save_screenshot(filename)
        self.query_one("#log", LogPanel).add_log(f"Screenshot: {filename}", "info")

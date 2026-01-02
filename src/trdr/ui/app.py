"""Main TUI application for trading bot."""

from datetime import datetime
from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header

from .messages import (
    LogMessage,
    MarketUpdate,
    PerformanceUpdate,
    PositionUpdate,
    StatusUpdate,
)
from .panels import (
    LogPanel,
    MarketPanel,
    PerformancePanel,
    PositionPanel,
    StatusBar,
)


class TradingBotApp(App):
    """Main TUI application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        height: 100%;
    }

    .panel {
        border: solid $primary;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    MarketPanel {
        height: 5;
    }

    PositionPanel {
        height: 5;
    }

    PerformancePanel {
        height: 5;
    }

    LogPanel {
        height: 12;
    }

    StatusBar {
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("p", "pause", "Pause"),
        ("r", "resume", "Resume"),
        ("s", "stop", "Stop"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self._paused = False
        self._on_pause: Callable | None = None
        self._on_resume: Callable | None = None
        self._on_stop: Callable | None = None

    def compose(self) -> ComposeResult:
        """Create the layout."""
        yield Header()
        with Container(id="main-container"):
            yield MarketPanel(classes="panel")
            yield PositionPanel(classes="panel")
            yield PerformancePanel(classes="panel")
            yield LogPanel(classes="panel")
            yield StatusBar()
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.title = "TRDR - Trading Bot"
        self.sub_title = "Volume Profile Strategy"
        self.query_one(LogPanel).add_log("Bot starting...", "info")

    # Message handlers
    def on_market_update(self, message: MarketUpdate) -> None:
        """Handle market data update."""
        panel = self.query_one(MarketPanel)
        panel.symbol = message.state.symbol
        panel.price = message.state.price
        panel.change_pct = message.state.change_pct
        panel.poc = message.state.poc
        panel.vah = message.state.vah
        panel.val = message.state.val

        # Update last update time
        status_bar = self.query_one(StatusBar)
        status_bar.last_update = datetime.now().strftime("%H:%M:%S")

    def on_position_update(self, message: PositionUpdate) -> None:
        """Handle position update."""
        panel = self.query_one(PositionPanel)
        panel.pos_side = message.state.side
        panel.pos_size = message.state.size
        panel.pos_entry = message.state.entry_price
        panel.pos_current = message.state.current_price
        panel.pos_pnl = message.state.unrealized_pnl
        panel.pos_stop = message.state.stop_loss or 0
        panel.pos_target = message.state.take_profit or 0

    def on_performance_update(self, message: PerformanceUpdate) -> None:
        """Handle performance update."""
        panel = self.query_one(PerformancePanel)
        panel.total_pnl = message.state.total_pnl
        panel.daily_pnl = message.state.daily_pnl
        panel.win_rate = message.state.win_rate
        panel.total_trades = message.state.total_trades
        panel.winning_trades = message.state.winning_trades

    def on_log_message(self, message: LogMessage) -> None:
        """Handle log message."""
        self.query_one(LogPanel).add_log(message.text, message.level)

    def on_status_update(self, message: StatusUpdate) -> None:
        """Handle status update."""
        self.query_one(StatusBar).status = message.status

    # Actions
    def action_pause(self) -> None:
        """Pause trading."""
        if not self._paused:
            self._paused = True
            self.query_one(StatusBar).status = "PAUSED"
            self.query_one(LogPanel).add_log("Trading paused by user", "warning")
            if self._on_pause:
                self._on_pause()

    def action_resume(self) -> None:
        """Resume trading."""
        if self._paused:
            self._paused = False
            self.query_one(StatusBar).status = "RUNNING"
            self.query_one(LogPanel).add_log("Trading resumed", "success")
            if self._on_resume:
                self._on_resume()

    def action_stop(self) -> None:
        """Stop trading and exit."""
        self.query_one(StatusBar).status = "STOPPED"
        self.query_one(LogPanel).add_log("Stopping bot...", "warning")
        if self._on_stop:
            self._on_stop()
        self.exit()

    def action_quit(self) -> None:
        """Quit application."""
        self.action_stop()

    def set_callbacks(
        self,
        on_pause: Callable | None = None,
        on_resume: Callable | None = None,
        on_stop: Callable | None = None,
    ) -> None:
        """Set callbacks for user actions.

        Args:
            on_pause: Called when user pauses
            on_resume: Called when user resumes
            on_stop: Called when user stops
        """
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_stop = on_stop

    @property
    def is_paused(self) -> bool:
        """Check if bot is paused."""
        return self._paused

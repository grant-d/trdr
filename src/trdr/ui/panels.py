"""TUI panel widgets for trading bot display."""

from datetime import datetime

from textual.reactive import reactive
from textual.widgets import Static


class MarketPanel(Static):
    """Displays current market data and Volume Profile levels."""

    symbol = reactive("---")
    price = reactive(0.0)
    change_pct = reactive(0.0)
    poc = reactive(0.0)
    vah = reactive(0.0)
    val = reactive(0.0)

    def render(self) -> str:
        sign = "+" if self.change_pct >= 0 else ""
        change_color = "green" if self.change_pct >= 0 else "red"

        return (
            f"[bold]MARKET DATA[/bold]\n"
            f"Symbol: [cyan]{self.symbol}[/]  "
            f"Price: [bold]${self.price:.2f}[/] "
            f"[{change_color}]({sign}{self.change_pct:.1f}%)[/]\n"
            f"POC: [yellow]${self.poc:.2f}[/]  "
            f"VA: ${self.val:.2f} - ${self.vah:.2f}"
        )


class PositionPanel(Static):
    """Displays current position details."""

    pos_side = reactive("NONE")
    pos_size = reactive(0.0)
    pos_entry = reactive(0.0)
    pos_current = reactive(0.0)
    pos_pnl = reactive(0.0)
    pos_stop = reactive(0.0)
    pos_target = reactive(0.0)

    def render(self) -> str:
        if self.pos_side == "NONE":
            return "[bold]POSITION[/bold]\n[dim]No open position[/]"

        pnl_color = "green" if self.pos_pnl >= 0 else "red"
        pnl_sign = "+" if self.pos_pnl >= 0 else ""

        stop_str = f"${self.pos_stop:.2f}" if self.pos_stop else "---"
        target_str = f"${self.pos_target:.2f}" if self.pos_target else "---"

        return (
            f"[bold]POSITION[/bold]\n"
            f"Side: [cyan]{self.pos_side}[/]  Size: {self.pos_size:.2f}  "
            f"Entry: ${self.pos_entry:.2f}\n"
            f"P&L: [{pnl_color}]{pnl_sign}${self.pos_pnl:.2f}[/]  "
            f"Stop: [red]{stop_str}[/]  Target: [green]{target_str}[/]"
        )


class PerformancePanel(Static):
    """Displays performance metrics."""

    total_pnl = reactive(0.0)
    daily_pnl = reactive(0.0)
    win_rate = reactive(0.0)
    total_trades = reactive(0)
    winning_trades = reactive(0)

    def render(self) -> str:
        total_color = "green" if self.total_pnl >= 0 else "red"
        daily_color = "green" if self.daily_pnl >= 0 else "red"
        total_sign = "+" if self.total_pnl >= 0 else ""
        daily_sign = "+" if self.daily_pnl >= 0 else ""

        return (
            f"[bold]PERFORMANCE[/bold]\n"
            f"Total P&L: [{total_color}]{total_sign}${self.total_pnl:.2f}[/]  "
            f"Today: [{daily_color}]{daily_sign}${self.daily_pnl:.2f}[/]\n"
            f"Win Rate: {self.win_rate:.1%} ({self.winning_trades}/{self.total_trades})"
        )


class LogPanel(Static):
    """Scrolling log of bot events."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logs: list[str] = []
        self._max_logs = 50

    def add_log(self, text: str, level: str = "info") -> None:
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {"info": "white", "warning": "yellow", "error": "red", "success": "green"}.get(
            level, "white"
        )
        entry = f"[dim]{timestamp}[/] [{color}]{text}[/]"
        self._logs.append(entry)

        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs :]

        self.refresh()

    def render(self) -> str:
        if not self._logs:
            return "[bold]LOGS[/bold]\n[dim]No events yet[/]"

        # Show last 8 logs
        recent = self._logs[-8:]
        return "[bold]LOGS[/bold]\n" + "\n".join(recent)


class StatusBar(Static):
    """Status bar showing bot state."""

    status = reactive("STARTING")
    last_update = reactive("")

    def render(self) -> str:
        status_colors = {
            "RUNNING": "green",
            "PAUSED": "yellow",
            "STOPPED": "red",
            "STARTING": "cyan",
            "ERROR": "red",
        }
        color = status_colors.get(self.status, "white")

        return f"[{color}]●[/] {self.status}  |  Last update: {self.last_update}"

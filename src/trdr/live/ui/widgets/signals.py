"""Signals panel widget."""

from dataclasses import dataclass
from datetime import datetime

from textual.widgets import Static


@dataclass
class SignalEntry:
    """A signal history entry."""

    timestamp: str
    action: str
    score: float
    reason: str


class SignalsPanel(Static):
    """Displays recent strategy signals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signals: list[SignalEntry] = []
        self._max_signals = 20

    def add_signal(self, action: str, score: float, reason: str = "") -> None:
        """Add a signal entry.

        Args:
            action: Signal action (LONG, SHORT, CLOSE, HOLD)
            score: Signal strength/confidence
            reason: Signal reason
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = SignalEntry(timestamp=timestamp, action=action, score=score, reason=reason)
        self._signals.append(entry)

        if len(self._signals) > self._max_signals:
            self._signals = self._signals[-self._max_signals :]

        self.refresh()

    def render(self) -> str:
        if not self._signals:
            return "[bold]RECENT SIGNALS[/]\n[dim]No signals yet[/]"

        lines = ["[bold]RECENT SIGNALS[/]"]
        for sig in reversed(self._signals[-5:]):
            action_colors = {
                "LONG": "cyan",
                "BUY": "cyan",
                "SHORT": "magenta",
                "SELL": "magenta",
                "CLOSE": "yellow",
                "HOLD": "dim",
            }
            color = action_colors.get(sig.action.upper(), "white")
            reason_str = f" {sig.reason}" if sig.reason else ""
            line = f"[dim]{sig.timestamp}[/] [{color}]{sig.action}[/] score={sig.score:.2f}"
            lines.append(f"{line}{reason_str}")

        return "\n".join(lines)

"""Log panel widget."""

from datetime import datetime

from textual.widgets import Static


class LogPanel(Static):
    """Scrolling log of harness events."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logs: list[str] = []
        self._max_logs = 100

    def add_log(self, text: str, level: str = "info") -> None:
        """Add a log entry.

        Args:
            text: Log message
            level: Log level (info, warning, error, success)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "info": "white",
            "warning": "yellow",
            "error": "red bold",
            "success": "green",
            "debug": "dim",
        }
        color = colors.get(level, "white")
        entry = f"[dim]{timestamp}[/] [{color}]{text}[/]"
        self._logs.append(entry)

        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs :]

        self.refresh()

    def render(self) -> str:
        if not self._logs:
            return "[bold]LOG[/]\n[dim]No events yet[/]"

        recent = self._logs[-10:]
        return "[bold]LOG[/]\n" + "\n".join(recent)

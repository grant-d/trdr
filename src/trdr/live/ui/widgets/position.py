"""Position panel widget."""

from textual.reactive import reactive
from textual.widgets import Static


class PositionPanel(Static):
    """Displays current position details."""

    side = reactive("FLAT")
    qty = reactive(0.0)
    entry = reactive(0.0)
    current = reactive(0.0)
    stop = reactive(0.0)
    target = reactive(0.0)

    @property
    def pnl(self) -> float:
        """Calculate unrealized PnL."""
        if self.side == "FLAT" or self.qty == 0:
            return 0.0
        if self.side == "LONG":
            return (self.current - self.entry) * self.qty
        return (self.entry - self.current) * self.qty

    @property
    def pnl_pct(self) -> float:
        """Calculate PnL percentage."""
        if self.entry == 0 or self.side == "FLAT":
            return 0.0
        if self.side == "LONG":
            return ((self.current - self.entry) / self.entry) * 100
        return ((self.entry - self.current) / self.entry) * 100

    def render(self) -> str:
        if self.side == "FLAT":
            return "[bold]POSITION[/]\n[dim]No open position[/]"

        pnl = self.pnl
        pnl_pct = self.pnl_pct
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""
        side_color = "cyan" if self.side == "LONG" else "magenta"

        stop_str = f"${self.stop:,.2f}" if self.stop else "---"
        target_str = f"${self.target:,.2f}" if self.target else "---"

        return (
            f"[bold]POSITION[/]\n"
            f"Side: [{side_color}]{self.side}[/]  Size: {self.qty:.4f}\n"
            f"Entry: ${self.entry:,.2f}  Current: ${self.current:,.2f}\n"
            f"PnL: [{pnl_color}]{pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)[/]\n"
            f"Stop: [red]{stop_str}[/]  Target: [green]{target_str}[/]"
        )

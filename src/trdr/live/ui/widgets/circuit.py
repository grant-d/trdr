"""Circuit breaker panel widget."""

from textual.reactive import reactive
from textual.widgets import Static


class CircuitPanel(Static):
    """Displays circuit breaker status and limits."""

    state = reactive("closed")
    equity = reactive(0.0)
    high_water_mark = reactive(0.0)
    drawdown_pct = reactive(0.0)
    daily_pnl = reactive(0.0)
    max_drawdown_pct = reactive(10.0)
    max_daily_loss = reactive(500.0)

    def render(self) -> str:
        state_colors = {
            "closed": "green",
            "open": "red bold",
            "half_open": "yellow",
        }
        state_color = state_colors.get(self.state.lower(), "white")

        dd_color = "green" if self.drawdown_pct < self.max_drawdown_pct * 0.5 else "yellow"
        if self.drawdown_pct >= self.max_drawdown_pct * 0.8:
            dd_color = "red"

        daily_color = "green" if self.daily_pnl >= 0 else "red"
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.max_daily_loss * 0.8:
            daily_color = "red bold"

        daily_sign = "+" if self.daily_pnl >= 0 else ""

        daily_limit = f"-${self.max_daily_loss:,.2f}"
        return (
            f"[bold]CIRCUIT BREAKER[/]\n"
            f"State: [{state_color}]{self.state.upper()}[/]\n"
            f"Equity: ${self.equity:,.2f}  HWM: ${self.high_water_mark:,.2f}\n"
            f"Drawdown: [{dd_color}]{self.drawdown_pct:.1f}%[/] / {self.max_drawdown_pct:.1f}%\n"
            f"Daily PnL: [{daily_color}]{daily_sign}${self.daily_pnl:,.2f}[/] / {daily_limit}"
        )

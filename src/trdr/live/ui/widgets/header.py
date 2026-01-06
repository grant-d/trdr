"""Status header widget."""

from textual.reactive import reactive
from textual.widgets import Static


class StatusHeader(Static):
    """Top header bar showing symbol, timeframe, state, and circuit status."""

    symbol = reactive("---")
    timeframe = reactive("---")
    state = reactive("INIT")
    circuit_state = reactive("CLOSED")
    mode = reactive("PAPER")
    price = reactive(0.0)
    bars_processed = reactive(0)

    def render(self) -> str:
        state_colors = {
            "RUNNING": "green",
            "PAUSED": "yellow",
            "STOPPED": "red",
            "INIT": "cyan",
            "ERROR": "red bold",
        }
        state_color = state_colors.get(self.state, "white")

        circuit_colors = {
            "closed": "green",
            "open": "red",
            "half_open": "yellow",
        }
        circuit_color = circuit_colors.get(self.circuit_state.lower(), "white")

        mode_color = "yellow" if self.mode == "PAPER" else "red"
        price_str = f"${self.price:,.2f}" if self.price else "---"

        return (
            f"[bold]TRDR Live[/] | "
            f"[cyan]{self.symbol}[/] | "  # Cyan for symbol
            f"{self.timeframe} | "
            f"[bold #00ff00]{price_str}[/] | "  # Bright green for price
            f"[{state_color}]{self.state}[/] | "
            f"Circuit: [{circuit_color}]{self.circuit_state.upper()}[/] | "
            f"[{mode_color}]{self.mode}[/] | "
            f"Bars: {self.bars_processed}"
        )

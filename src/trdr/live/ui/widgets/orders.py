"""Orders panel widget."""

from textual.widgets import Static


class OrdersPanel(Static):
    """Displays pending orders."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orders: list[dict] = []

    def update_orders(self, orders: list[dict]) -> None:
        """Update orders list.

        Args:
            orders: List of order dicts with keys: side, type, qty, price, symbol
        """
        self._orders = orders
        self.refresh()

    def render(self) -> str:
        if not self._orders:
            return "[bold]PENDING ORDERS[/]\n[dim]No pending orders[/]"

        lines = ["[bold]PENDING ORDERS[/]"]
        for order in self._orders[:5]:  # Show max 5
            side = order.get("side", "?")
            order_type = order.get("type", "?")
            qty = order.get("qty", 0)
            price = order.get("price")

            side_color = "green" if side.upper() == "BUY" else "red"
            price_str = f"@ ${price:,.2f}" if price else "@ MARKET"

            lines.append(f"[{side_color}]{side.upper()}[/] {order_type} {qty:.4f} {price_str}")

        if len(self._orders) > 5:
            lines.append(f"[dim]+{len(self._orders) - 5} more[/]")

        return "\n".join(lines)

"""Tests for order management."""

import pytest

from trdr.backtest.orders import Fill, Order, OrderManager, OrderType
from trdr.data.market import Bar


def make_bar(
    timestamp: str = "2024-01-01T10:00:00Z",
    open: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: int = 1000,
) -> Bar:
    """Create test bar."""
    return Bar(
        timestamp=timestamp,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class TestOrder:
    """Tests for Order dataclass."""

    def test_market_order_no_stop_required(self) -> None:
        """Market orders don't require stop_price."""
        order = Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=10.0,
        )
        assert order.status == "pending"
        assert order.stop_price is None

    def test_stop_order_requires_stop_price(self) -> None:
        """Stop orders require stop_price."""
        with pytest.raises(ValueError, match="requires stop_price"):
            Order(
                symbol="TEST",
                side="buy",
                order_type=OrderType.STOP,
                quantity=10.0,
            )

    def test_stop_loss_requires_stop_price(self) -> None:
        """Stop loss orders require stop_price."""
        with pytest.raises(ValueError, match="requires stop_price"):
            Order(
                symbol="TEST",
                side="sell",
                order_type=OrderType.STOP_LOSS,
                quantity=10.0,
            )

    def test_trailing_stop_requires_trail(self) -> None:
        """Trailing stop requires trail_percent or trail_amount."""
        with pytest.raises(ValueError, match="trail_percent or trail_amount"):
            Order(
                symbol="TEST",
                side="sell",
                order_type=OrderType.TRAILING_STOP,
                quantity=10.0,
            )

    def test_trailing_stop_with_percent(self) -> None:
        """Trailing stop with percent is valid."""
        order = Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP,
            quantity=10.0,
            trail_percent=0.02,
        )
        assert order.trail_percent == 0.02

    def test_trailing_stop_with_amount(self) -> None:
        """Trailing stop with amount is valid."""
        order = Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP,
            quantity=10.0,
            trail_amount=5.0,
        )
        assert order.trail_amount == 5.0


class TestOrderManager:
    """Tests for OrderManager."""

    def test_submit_order(self) -> None:
        """Submit adds order to pending."""
        manager = OrderManager()
        order = Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=10.0,
        )
        order_id = manager.submit(order)
        assert len(manager.pending_orders) == 1
        assert manager.pending_orders[0].id == order_id

    def test_cancel_order(self) -> None:
        """Cancel removes order from pending."""
        manager = OrderManager()
        order = Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=10.0,
        )
        order_id = manager.submit(order)
        assert manager.cancel(order_id)
        assert len(manager.pending_orders) == 0

    def test_cancel_nonexistent(self) -> None:
        """Cancel returns False for unknown order."""
        manager = OrderManager()
        assert not manager.cancel("nonexistent")

    def test_cancel_all(self) -> None:
        """Cancel all clears pending orders."""
        manager = OrderManager()
        manager.submit(Order(symbol="A", side="buy", order_type=OrderType.MARKET, quantity=1))
        manager.submit(Order(symbol="B", side="buy", order_type=OrderType.MARKET, quantity=1))
        count = manager.cancel_all()
        assert count == 2
        assert len(manager.pending_orders) == 0

    def test_market_order_fills_at_open(self) -> None:
        """Market order fills at bar open."""
        manager = OrderManager()
        order = Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=10.0,
        )
        manager.submit(order)

        bar = make_bar(open=100.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 100.0
        assert fills[0].quantity == 10.0
        assert len(manager.pending_orders) == 0

    def test_market_order_with_slippage(self) -> None:
        """Market buy order has slippage added."""
        manager = OrderManager()
        manager.submit(Order(symbol="TEST", side="buy", order_type=OrderType.MARKET, quantity=10))

        bar = make_bar(open=100.0)
        fills = manager.process_bar(bar, slippage=0.5)

        assert fills[0].price == 100.5  # Buy pays more

    def test_market_sell_slippage(self) -> None:
        """Market sell order has slippage subtracted."""
        manager = OrderManager()
        manager.submit(Order(symbol="TEST", side="sell", order_type=OrderType.MARKET, quantity=10))

        bar = make_bar(open=100.0)
        fills = manager.process_bar(bar, slippage=0.5)

        assert fills[0].price == 99.5  # Sell receives less

    def test_stop_order_triggers(self) -> None:
        """Stop buy triggers when price rises above stop."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=103.0,
        ))

        # Bar with high above stop
        bar = make_bar(open=100.0, high=105.0, low=99.0, close=104.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 103.0  # Fills at stop price

    def test_stop_order_not_triggered(self) -> None:
        """Stop buy doesn't trigger if price stays below."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=110.0,
        ))

        bar = make_bar(open=100.0, high=105.0, low=99.0, close=102.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 0
        assert len(manager.pending_orders) == 1

    def test_stop_loss_triggers(self) -> None:
        """Stop loss (sell) triggers when price drops below."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP_LOSS,
            quantity=10,
            stop_price=97.0,
        ))

        bar = make_bar(open=100.0, high=101.0, low=95.0, close=96.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 97.0

    def test_stop_loss_gap_down(self) -> None:
        """Stop loss fills at open if price gaps through."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP_LOSS,
            quantity=10,
            stop_price=100.0,
        ))

        # Gap down - open below stop
        bar = make_bar(open=95.0, high=96.0, low=94.0, close=95.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 95.0  # Fills at open (worse than stop)


class TestTrailingStop:
    """Tests for trailing stop orders."""

    def test_tsl_moves_up_with_price(self) -> None:
        """Long TSL moves up as price rises."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP,
            quantity=10,
            trail_percent=0.02,  # 2% trail
        ))

        # First bar - set initial stop
        bar1 = make_bar(high=100.0)
        manager.update_trailing_stops(bar1)
        assert manager.pending_orders[0].stop_price == 98.0  # 100 * 0.98

        # Price rises - stop moves up
        bar2 = make_bar(high=110.0)
        manager.update_trailing_stops(bar2)
        assert manager.pending_orders[0].stop_price == 107.8  # 110 * 0.98

    def test_tsl_never_moves_down(self) -> None:
        """Long TSL never moves down."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP,
            quantity=10,
            trail_percent=0.02,
        ))

        bar1 = make_bar(high=100.0)
        manager.update_trailing_stops(bar1)
        stop_after_first = manager.pending_orders[0].stop_price

        # Price drops - stop stays same
        bar2 = make_bar(high=95.0)
        manager.update_trailing_stops(bar2)
        assert manager.pending_orders[0].stop_price == stop_after_first

    def test_tsl_triggers_on_reversal(self) -> None:
        """TSL triggers when price drops to stop."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP,
            quantity=10,
            trail_amount=5.0,  # $5 trail
        ))

        # Set initial stop
        bar1 = make_bar(high=100.0)
        manager.update_trailing_stops(bar1)
        assert manager.pending_orders[0].stop_price == 95.0

        # Price drops through stop
        bar2 = make_bar(open=96.0, high=97.0, low=93.0, close=94.0)
        manager.update_trailing_stops(bar2)  # Stop stays at 95
        fills = manager.process_bar(bar2)

        assert len(fills) == 1
        assert fills[0].price == 95.0

    def test_short_tsl_moves_down(self) -> None:
        """Short TSL moves down as price drops."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",  # Buy to cover short
            order_type=OrderType.TRAILING_STOP,
            quantity=10,
            trail_percent=0.02,
        ))

        bar1 = make_bar(low=100.0)
        manager.update_trailing_stops(bar1)
        assert manager.pending_orders[0].stop_price == 102.0  # 100 * 1.02

        # Price drops - stop moves down
        bar2 = make_bar(low=90.0)
        manager.update_trailing_stops(bar2)
        assert manager.pending_orders[0].stop_price == 91.8  # 90 * 1.02

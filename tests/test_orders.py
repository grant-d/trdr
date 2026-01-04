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

    def test_sell_stop_triggers_on_drop(self) -> None:
        """Sell stop triggers when price drops to stop."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=97.0,
        ))

        bar = make_bar(open=100.0, high=101.0, low=95.0, close=96.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 97.0

    def test_sell_stop_triggers_on_rise(self) -> None:
        """Sell stop (take profit) triggers when price rises to stop."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=108.0,
        ))

        bar = make_bar(open=100.0, high=110.0, low=99.0, close=109.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 108.0  # Fills at stop price

    def test_stop_fills_at_stop_price(self) -> None:
        """Stop fills at stop price when touched during bar."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=95.0,
        ))

        # Bar range includes stop price
        bar = make_bar(open=100.0, high=101.0, low=94.0, close=96.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 95.0  # Fills at stop price

    def test_stop_not_triggered_outside_range(self) -> None:
        """Stop doesn't trigger if price never touches stop."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=90.0,
        ))

        # Bar range doesn't include stop price
        bar = make_bar(open=100.0, high=105.0, low=95.0, close=102.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 0
        assert len(manager.pending_orders) == 1


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


class TestStopAbovePrice:
    """Tests for stop orders above current price (take profit behavior)."""

    def test_sell_stop_above_triggers_on_rise(self) -> None:
        """Sell stop above price triggers when price rises to target."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=108.0,
        ))

        # Bar with high above target
        bar = make_bar(open=100.0, high=110.0, low=99.0, close=109.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 108.0  # Fills at stop price

    def test_sell_stop_above_not_triggered_below(self) -> None:
        """Sell stop above price doesn't trigger if price stays below."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=120.0,
        ))

        bar = make_bar(open=100.0, high=110.0, low=99.0, close=108.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 0
        assert len(manager.pending_orders) == 1

    def test_sell_stop_above_fills_at_stop(self) -> None:
        """Sell stop above price fills at stop when touched."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=112.0,
        ))

        # Bar range includes stop
        bar = make_bar(open=110.0, high=115.0, low=108.0, close=114.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 112.0  # Fills at stop price

    def test_buy_stop_below_triggers_on_drop(self) -> None:
        """Buy stop below price (short cover) triggers when price drops."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=90.0,
        ))

        bar = make_bar(open=100.0, high=101.0, low=88.0, close=89.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 90.0  # Fills at stop


class TestTrailingStopLimit:
    """Tests for trailing stop-limit orders."""

    def test_trailing_stop_limit_requires_trail_and_limit(self) -> None:
        """Trailing stop-limit requires trail and limit_price."""
        with pytest.raises(ValueError, match="trail_percent or trail_amount"):
            Order(
                symbol="TEST",
                side="sell",
                order_type=OrderType.TRAILING_STOP_LIMIT,
                quantity=10.0,
                limit_price=95.0,
            )

        with pytest.raises(ValueError, match="requires limit_price"):
            Order(
                symbol="TEST",
                side="sell",
                order_type=OrderType.TRAILING_STOP_LIMIT,
                quantity=10.0,
                trail_percent=0.02,
            )

    def test_trailing_stop_limit_valid(self) -> None:
        """Trailing stop-limit with trail and limit is valid."""
        order = Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP_LIMIT,
            quantity=10.0,
            trail_percent=0.02,
            limit_price=95.0,
        )
        assert order.trail_percent == 0.02
        assert order.limit_price == 95.0

    def test_tsl_limit_trails_then_fills_at_limit(self) -> None:
        """TSL-limit trails stop, then fills at limit price when triggered."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP_LIMIT,
            quantity=10,
            trail_percent=0.02,  # 2% trail
            limit_price=105.0,  # Fill at 105 or better
        ))

        # Bar 1: Set initial trailing stop at 98 (100 * 0.98)
        bar1 = make_bar(open=100.0, high=100.0, low=99.0, close=100.0)
        manager.update_trailing_stops(bar1)
        assert manager.pending_orders[0].stop_price == 98.0

        # Bar 2: Price rises, stop trails up to 107.8 (110 * 0.98)
        # Bar stays above trailed stop so no trigger
        bar2 = make_bar(open=109.0, high=110.0, low=108.0, close=109.0)
        manager.update_trailing_stops(bar2)
        fills = manager.process_bar(bar2)
        assert len(fills) == 0
        assert manager.pending_orders[0].stop_price == 107.8

        # Bar 3: Price drops, touches stop (107.8), limit fills at 105
        # Open at 100, which is below limit, so fills at limit
        bar3 = make_bar(open=100.0, high=108.0, low=95.0, close=96.0)
        manager.update_trailing_stops(bar3)  # Stop stays at 107.8
        fills = manager.process_bar(bar3)

        assert len(fills) == 1
        assert fills[0].price == 105.0  # Fills at limit price

    def test_tsl_limit_triggered_but_limit_not_filled(self) -> None:
        """TSL-limit can trigger but not fill if limit not reached."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP_LIMIT,
            quantity=10,
            trail_percent=0.02,
            limit_price=110.0,  # High limit price
        ))

        # Set initial stop
        bar1 = make_bar(high=100.0)
        manager.update_trailing_stops(bar1)

        # Trigger stop but limit not reached
        bar2 = make_bar(open=99.0, high=99.0, low=97.0, close=97.5)
        manager.update_trailing_stops(bar2)
        fills = manager.process_bar(bar2)

        assert len(fills) == 0
        assert manager.pending_orders[0].triggered is True  # Triggered
        assert len(manager.pending_orders) == 1  # But not filled

    def test_tsl_limit_triggered_fills_on_later_bar(self) -> None:
        """TSL-limit triggers, hangs around, then fills on later bar."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.TRAILING_STOP_LIMIT,
            quantity=10,
            trail_percent=0.02,
            limit_price=105.0,  # Need price to rise to 105 to fill
        ))

        # Bar 1: Set initial stop at 98 (100 * 0.98)
        bar1 = make_bar(open=100.0, high=100.0, low=99.0, close=100.0)
        manager.update_trailing_stops(bar1)
        assert manager.pending_orders[0].stop_price == 98.0

        # Bar 2: Triggers stop (low=97 touches 98) but limit not reached (high=99 < 105)
        bar2 = make_bar(open=99.0, high=99.0, low=97.0, close=98.0)
        manager.update_trailing_stops(bar2)
        fills = manager.process_bar(bar2)
        assert len(fills) == 0
        assert manager.pending_orders[0].triggered is True

        # Bar 3: Still no fill (high=102 < 105)
        bar3 = make_bar(open=100.0, high=102.0, low=99.0, close=101.0)
        fills = manager.process_bar(bar3)
        assert len(fills) == 0
        assert len(manager.pending_orders) == 1  # Still hanging around

        # Bar 4: Finally fills when high reaches limit
        bar4 = make_bar(open=104.0, high=107.0, low=103.0, close=106.0)
        fills = manager.process_bar(bar4)
        assert len(fills) == 1
        assert fills[0].price == 105.0  # Fills at limit price


class TestLimitOrders:
    """Tests for limit orders."""

    def test_limit_order_requires_limit_price(self) -> None:
        """Limit orders require limit_price."""
        with pytest.raises(ValueError, match="requires limit_price"):
            Order(
                symbol="TEST",
                side="buy",
                order_type=OrderType.LIMIT,
                quantity=10.0,
            )

    def test_limit_order_valid_with_limit_price(self) -> None:
        """Limit order with limit_price is valid."""
        order = Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=10.0,
            limit_price=95.0,
        )
        assert order.limit_price == 95.0
        assert order.order_type == OrderType.LIMIT

    def test_buy_limit_triggers_on_price_drop(self) -> None:
        """Buy limit triggers when price drops to limit."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=97.0,
        ))

        # Bar with low at/below limit
        bar = make_bar(open=100.0, high=101.0, low=95.0, close=96.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 97.0  # Fills at limit price

    def test_buy_limit_not_triggered_above_limit(self) -> None:
        """Buy limit doesn't trigger if price stays above limit."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=90.0,
        ))

        bar = make_bar(open=100.0, high=105.0, low=98.0, close=102.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 0
        assert len(manager.pending_orders) == 1

    def test_buy_limit_gap_down(self) -> None:
        """Buy limit fills at open if price gaps through limit."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=98.0,
        ))

        # Gap down - open below limit (better price)
        bar = make_bar(open=95.0, high=96.0, low=94.0, close=95.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 95.0  # Fills at open (better than limit)

    def test_sell_limit_triggers_on_price_rise(self) -> None:
        """Sell limit triggers when price rises to limit."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=108.0,
        ))

        bar = make_bar(open=100.0, high=110.0, low=99.0, close=109.0)
        fills = manager.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].price == 108.0  # Fills at limit price

    def test_limit_order_no_slippage(self) -> None:
        """Limit orders have no slippage applied."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=97.0,
        ))

        bar = make_bar(open=100.0, high=101.0, low=95.0, close=96.0)
        fills = manager.process_bar(bar, slippage=0.5)  # Slippage should be ignored

        assert len(fills) == 1
        assert fills[0].price == 97.0  # No slippage added


class TestStopLimitOrders:
    """Tests for stop-limit orders."""

    def test_stop_limit_requires_both_prices(self) -> None:
        """Stop-limit orders require both stop_price and limit_price."""
        with pytest.raises(ValueError, match="requires both stop_price and limit_price"):
            Order(
                symbol="TEST",
                side="buy",
                order_type=OrderType.STOP_LIMIT,
                quantity=10.0,
                stop_price=105.0,
                # missing limit_price
            )

        with pytest.raises(ValueError, match="requires both stop_price and limit_price"):
            Order(
                symbol="TEST",
                side="buy",
                order_type=OrderType.STOP_LIMIT,
                quantity=10.0,
                limit_price=106.0,
                # missing stop_price
            )

    def test_stop_limit_valid_with_both_prices(self) -> None:
        """Stop-limit with both prices is valid."""
        order = Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.STOP_LIMIT,
            quantity=10.0,
            stop_price=105.0,
            limit_price=106.0,
        )
        assert order.stop_price == 105.0
        assert order.limit_price == 106.0
        assert order.triggered is False

    def test_buy_stop_limit_triggers_then_fills(self) -> None:
        """Buy stop-limit triggers on breakout, fills at limit."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.STOP_LIMIT,
            quantity=10,
            stop_price=105.0,  # Trigger when price rises to 105
            limit_price=106.0,  # Fill at 106 or better
        ))

        # Bar 1: Price below stop - not triggered
        bar1 = make_bar(open=100.0, high=103.0, low=99.0, close=102.0)
        fills = manager.process_bar(bar1)
        assert len(fills) == 0
        assert manager.pending_orders[0].triggered is False

        # Bar 2: Price hits stop (high=107) and limit (open=104 < 106)
        bar2 = make_bar(open=104.0, high=107.0, low=103.0, close=106.0)
        fills = manager.process_bar(bar2)
        assert len(fills) == 1
        assert fills[0].price == 104.0  # Fills at open (better than limit)

    def test_buy_stop_limit_triggers_but_no_fill(self) -> None:
        """Buy stop-limit triggers but limit never fills."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.STOP_LIMIT,
            quantity=10,
            stop_price=105.0,
            limit_price=103.0,  # Want to buy at 103 or lower
        ))

        # Bar 1: Triggers stop (high=107) but never hits limit (low=104)
        bar1 = make_bar(open=106.0, high=110.0, low=104.0, close=108.0)
        fills = manager.process_bar(bar1)
        assert len(fills) == 0
        assert manager.pending_orders[0].triggered is True  # Triggered but not filled

        # Bar 2: Still above limit
        bar2 = make_bar(open=107.0, high=109.0, low=105.0, close=108.0)
        fills = manager.process_bar(bar2)
        assert len(fills) == 0

        # Bar 3: Finally hits limit
        bar3 = make_bar(open=104.0, high=105.0, low=102.0, close=103.0)
        fills = manager.process_bar(bar3)
        assert len(fills) == 1
        assert fills[0].price == 103.0  # Fills at limit

    def test_sell_stop_limit_triggers_then_fills(self) -> None:
        """Sell stop-limit triggers on breakdown, fills at limit."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="sell",
            order_type=OrderType.STOP_LIMIT,
            quantity=10,
            stop_price=95.0,  # Trigger when price drops to 95
            limit_price=94.0,  # Fill at 94 or better
        ))

        # Bar 1: Price above stop - not triggered
        bar1 = make_bar(open=100.0, high=102.0, low=97.0, close=98.0)
        fills = manager.process_bar(bar1)
        assert len(fills) == 0

        # Bar 2: Price hits stop and limit
        bar2 = make_bar(open=96.0, high=97.0, low=93.0, close=94.0)
        fills = manager.process_bar(bar2)
        assert len(fills) == 1
        assert fills[0].price == 96.0  # Fills at open (better than limit)

    def test_stop_limit_no_slippage(self) -> None:
        """Stop-limit orders have no slippage."""
        manager = OrderManager()
        manager.submit(Order(
            symbol="TEST",
            side="buy",
            order_type=OrderType.STOP_LIMIT,
            quantity=10,
            stop_price=105.0,
            limit_price=106.0,
        ))

        # Triggers and fills with slippage param - should be ignored
        bar = make_bar(open=104.0, high=107.0, low=103.0, close=106.0)
        fills = manager.process_bar(bar, slippage=1.0)

        assert len(fills) == 1
        assert fills[0].price == 104.0  # No slippage added

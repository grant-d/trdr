"""Tests for live trading types."""

from datetime import UTC, datetime

import pytest

from trdr.live.exchange.types import (
    HydraAccountInfo,
    HydraBar,
    HydraOrderRequest,
    HydraOrderResponse,
    HydraOrderSide,
    HydraOrderStatus,
    HydraOrderType,
    HydraPositionInfo,
)


class TestHydraBar:
    """Tests for HydraBar."""

    def test_create_bar(self):
        """Test bar creation."""
        bar = HydraBar(
            timestamp="2024-01-01T00:00:00Z",
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        )
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 99.0
        assert bar.close == 103.0
        assert bar.volume == 1000.0

    def test_bar_is_frozen(self):
        """Test bar is immutable."""
        bar = HydraBar(
            timestamp="2024-01-01T00:00:00Z",
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        )
        with pytest.raises(Exception):
            bar.close = 110.0


class TestHydraOrderRequest:
    """Tests for HydraOrderRequest."""

    def test_market_order_defaults(self):
        """Test market order defaults."""
        order = HydraOrderRequest(
            symbol="ETH/USD",
            side=HydraOrderSide.BUY,
            qty=1.0,
        )
        assert order.order_type == HydraOrderType.MARKET
        assert order.limit_price is None
        assert order.stop_price is None
        assert order.time_in_force == "gtc"
        assert len(order.client_order_id) == 12

    def test_limit_order(self):
        """Test limit order creation."""
        order = HydraOrderRequest(
            symbol="ETH/USD",
            side=HydraOrderSide.BUY,
            qty=1.0,
            order_type=HydraOrderType.LIMIT,
            limit_price=100.0,
        )
        assert order.order_type == HydraOrderType.LIMIT
        assert order.limit_price == 100.0

    def test_stop_limit_order(self):
        """Test stop-limit order creation."""
        order = HydraOrderRequest(
            symbol="ETH/USD",
            side=HydraOrderSide.SELL,
            qty=1.0,
            order_type=HydraOrderType.STOP_LIMIT,
            stop_price=95.0,
            limit_price=94.0,
        )
        assert order.order_type == HydraOrderType.STOP_LIMIT
        assert order.stop_price == 95.0
        assert order.limit_price == 94.0


class TestHydraOrderResponse:
    """Tests for HydraOrderResponse."""

    def test_order_response(self):
        """Test order response creation."""
        now = datetime.now(UTC)
        response = HydraOrderResponse(
            order_id="abc123",
            client_order_id="client123",
            symbol="ETH/USD",
            side=HydraOrderSide.BUY,
            qty=1.0,
            filled_qty=0.5,
            filled_avg_price=100.0,
            status=HydraOrderStatus.PARTIALLY_FILLED,
            order_type=HydraOrderType.LIMIT,
            submitted_at=now,
        )
        assert response.order_id == "abc123"
        assert response.filled_qty == 0.5
        assert response.status == HydraOrderStatus.PARTIALLY_FILLED


class TestHydraPositionInfo:
    """Tests for HydraPositionInfo."""

    def test_long_position(self):
        """Test long position creation."""
        pos = HydraPositionInfo(
            symbol="ETH/USD",
            side="long",
            qty=1.0,
            avg_entry_price=100.0,
            market_value=105.0,
            unrealized_pnl=5.0,
            qty_available=1.0,
        )
        assert pos.side == "long"
        assert pos.qty == 1.0
        assert pos.unrealized_pnl == 5.0

    def test_short_position(self):
        """Test short position creation."""
        pos = HydraPositionInfo(
            symbol="ETH/USD",
            side="short",
            qty=1.0,
            avg_entry_price=100.0,
            market_value=95.0,
            unrealized_pnl=5.0,
            qty_available=1.0,
        )
        assert pos.side == "short"


class TestHydraAccountInfo:
    """Tests for HydraAccountInfo."""

    def test_account_info(self):
        """Test account info creation."""
        account = HydraAccountInfo(
            equity=10000.0,
            cash=5000.0,
            buying_power=20000.0,
        )
        assert account.equity == 10000.0
        assert account.cash == 5000.0
        assert account.buying_power == 20000.0
        assert account.currency == "USD"

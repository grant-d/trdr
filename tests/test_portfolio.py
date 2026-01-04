"""Tests for portfolio and position tracking."""

import pytest

from trdr.backtest.portfolio import Portfolio, Position, PositionEntry


class TestPositionEntry:
    """Tests for PositionEntry."""

    def test_create_entry(self) -> None:
        """Create position entry."""
        entry = PositionEntry(price=100.0, quantity=10.0, timestamp="2024-01-01T10:00:00Z")
        assert entry.price == 100.0
        assert entry.quantity == 10.0


class TestPosition:
    """Tests for Position."""

    def test_empty_position(self) -> None:
        """Empty position has zero quantity."""
        pos = Position(symbol="TEST", side="long")
        assert pos.total_quantity == 0
        assert pos.avg_price == 0.0

    def test_single_entry(self) -> None:
        """Single entry position."""
        pos = Position(symbol="TEST", side="long")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")
        assert pos.total_quantity == 10.0
        assert pos.avg_price == 100.0

    def test_multiple_entries_equal(self) -> None:
        """Multiple entries with same price."""
        pos = Position(symbol="TEST", side="long")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")
        pos.add_entry(100.0, 10.0, "2024-01-01T11:00:00Z")
        assert pos.total_quantity == 20.0
        assert pos.avg_price == 100.0

    def test_multiple_entries_different_prices(self) -> None:
        """Multiple entries average correctly."""
        pos = Position(symbol="TEST", side="long")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")  # $1000
        pos.add_entry(110.0, 10.0, "2024-01-01T11:00:00Z")  # $1100
        assert pos.total_quantity == 20.0
        assert pos.avg_price == 105.0  # $2100 / 20

    def test_unrealized_pnl_long_profit(self) -> None:
        """Long position unrealized profit."""
        pos = Position(symbol="TEST", side="long")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")
        pnl = pos.unrealized_pnl(current_price=110.0)
        assert pnl == 100.0  # (110 - 100) * 10

    def test_unrealized_pnl_long_loss(self) -> None:
        """Long position unrealized loss."""
        pos = Position(symbol="TEST", side="long")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")
        pnl = pos.unrealized_pnl(current_price=90.0)
        assert pnl == -100.0  # (90 - 100) * 10

    def test_unrealized_pnl_short_profit(self) -> None:
        """Short position unrealized profit."""
        pos = Position(symbol="TEST", side="short")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")
        pnl = pos.unrealized_pnl(current_price=90.0)
        assert pnl == 100.0  # (100 - 90) * 10

    def test_reduce_full(self) -> None:
        """Reduce entire position."""
        pos = Position(symbol="TEST", side="long")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")
        avg = pos.reduce(10.0)
        assert avg == 100.0
        assert pos.total_quantity == 0

    def test_reduce_partial_fifo(self) -> None:
        """Partial reduction uses FIFO."""
        pos = Position(symbol="TEST", side="long")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")
        pos.add_entry(120.0, 10.0, "2024-01-01T11:00:00Z")

        # Remove 15 units - should take all of first (10 @ 100) + 5 of second (5 @ 120)
        avg = pos.reduce(15.0)
        expected_avg = (100.0 * 10 + 120.0 * 5) / 15
        assert abs(avg - expected_avg) < 0.01
        assert pos.total_quantity == 5.0
        assert len(pos.entries) == 1
        assert pos.entries[0].quantity == 5.0  # Remaining from second entry

    def test_reduce_more_than_available(self) -> None:
        """Reduce capped at available quantity."""
        pos = Position(symbol="TEST", side="long")
        pos.add_entry(100.0, 10.0, "2024-01-01T10:00:00Z")
        avg = pos.reduce(100.0)  # Request more than available
        assert avg == 100.0
        assert pos.total_quantity == 0


class TestPortfolio:
    """Tests for Portfolio."""

    def test_initial_cash(self) -> None:
        """Portfolio starts with cash."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.cash == 10000.0
        assert portfolio.buying_power() == 10000.0

    def test_equity_no_positions(self) -> None:
        """Equity equals cash when no positions."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.equity({}) == 10000.0

    def test_open_position(self) -> None:
        """Open position reduces cash."""
        portfolio = Portfolio(cash=10000.0)
        portfolio.open_position("TEST", "long", 100.0, 50.0, "2024-01-01T10:00:00Z")
        assert portfolio.cash == 5000.0  # 10000 - (100 * 50)
        assert portfolio.get_position("TEST") is not None
        assert portfolio.get_position("TEST").total_quantity == 50.0

    def test_open_position_with_cost(self) -> None:
        """Open position with transaction cost."""
        portfolio = Portfolio(cash=10000.0)
        portfolio.open_position("TEST", "long", 100.0, 50.0, "2024-01-01T10:00:00Z", cost=25.0)
        assert portfolio.cash == 4975.0  # 10000 - (100 * 50) - 25

    def test_add_to_position(self) -> None:
        """Add to existing position."""
        portfolio = Portfolio(cash=10000.0)
        portfolio.open_position("TEST", "long", 100.0, 30.0, "2024-01-01T10:00:00Z")
        portfolio.open_position("TEST", "long", 110.0, 20.0, "2024-01-01T11:00:00Z")

        pos = portfolio.get_position("TEST")
        assert pos.total_quantity == 50.0
        assert pos.avg_price == 104.0  # (100*30 + 110*20) / 50
        assert portfolio.cash == 4800.0  # 10000 - 3000 - 2200

    def test_close_position_profit(self) -> None:
        """Close position with profit."""
        portfolio = Portfolio(cash=10000.0)
        portfolio.open_position("TEST", "long", 100.0, 50.0, "2024-01-01T10:00:00Z")
        pnl = portfolio.close_position("TEST", 110.0)

        assert pnl == 500.0  # (110 - 100) * 50
        assert portfolio.cash == 10500.0  # 5000 + 5500
        assert portfolio.get_position("TEST") is None

    def test_close_position_loss(self) -> None:
        """Close position with loss."""
        portfolio = Portfolio(cash=10000.0)
        portfolio.open_position("TEST", "long", 100.0, 50.0, "2024-01-01T10:00:00Z")
        pnl = portfolio.close_position("TEST", 90.0)

        assert pnl == -500.0  # (90 - 100) * 50
        assert portfolio.cash == 9500.0  # 5000 + 4500

    def test_close_position_with_cost(self) -> None:
        """Close position with transaction cost."""
        portfolio = Portfolio(cash=10000.0)
        portfolio.open_position("TEST", "long", 100.0, 50.0, "2024-01-01T10:00:00Z")
        pnl = portfolio.close_position("TEST", 110.0, cost=25.0)

        assert pnl == 475.0  # (110 - 100) * 50 - 25
        assert portfolio.cash == 10475.0

    def test_partial_close(self) -> None:
        """Partial position close."""
        portfolio = Portfolio(cash=10000.0)
        portfolio.open_position("TEST", "long", 100.0, 50.0, "2024-01-01T10:00:00Z")
        pnl = portfolio.close_position("TEST", 110.0, quantity=20.0)

        assert pnl == 200.0  # (110 - 100) * 20
        assert portfolio.get_position("TEST").total_quantity == 30.0
        assert portfolio.cash == 7200.0  # 5000 + 2200

    def test_equity_with_unrealized(self) -> None:
        """Equity includes unrealized P&L."""
        portfolio = Portfolio(cash=10000.0)
        portfolio.open_position("TEST", "long", 100.0, 50.0, "2024-01-01T10:00:00Z")

        # Price up 10%
        equity = portfolio.equity({"TEST": 110.0})
        assert equity == 10500.0  # 5000 cash + 5500 position value

        # Price down 10%
        equity = portfolio.equity({"TEST": 90.0})
        assert equity == 9500.0  # 5000 cash + 4500 position value

    def test_liquidate(self) -> None:
        """Liquidate generates close orders for all positions."""
        portfolio = Portfolio(cash=5000.0)
        portfolio.open_position("A", "long", 100.0, 10.0, "2024-01-01T10:00:00Z")
        portfolio.open_position("B", "long", 50.0, 20.0, "2024-01-01T10:00:00Z")

        orders = portfolio.liquidate({"A": 100.0, "B": 50.0})
        assert len(orders) == 2
        assert all(o.side == "sell" for o in orders)
        assert {o.symbol for o in orders} == {"A", "B"}

    def test_position_capped_by_buying_power(self) -> None:
        """Position size limited by available cash."""
        portfolio = Portfolio(cash=1000.0)
        # Try to buy $2000 worth
        portfolio.open_position("TEST", "long", 100.0, 20.0, "2024-01-01T10:00:00Z")

        pos = portfolio.get_position("TEST")
        assert pos.total_quantity == 10.0  # Capped at $1000 / $100
        assert portfolio.cash == 0.0

    def test_get_position_empty(self) -> None:
        """Get position returns None for non-existent."""
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.get_position("NONEXISTENT") is None

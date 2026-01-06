"""Tests for circuit breaker."""

from trdr.live.config import RiskLimits
from trdr.live.safety.circuit_breaker import (
    BreakerState,
    CircuitBreaker,
    TradingStats,
)


class TestTradingStats:
    """Tests for TradingStats."""

    def test_initial_state(self):
        """Test initial stats state."""
        stats = TradingStats()
        assert stats.start_equity == 0.0
        assert stats.high_water_mark == 0.0
        assert stats.current_equity == 0.0
        assert stats.daily_pnl == 0.0
        assert stats.consecutive_losses == 0

    def test_drawdown_pct(self):
        """Test drawdown percentage calculation."""
        stats = TradingStats(
            start_equity=10000.0,
            high_water_mark=11000.0,
            current_equity=10000.0,
        )
        # Drawdown = (11000 - 10000) / 11000 = 9.09%
        assert abs(stats.drawdown_pct - 9.09) < 0.1

    def test_daily_loss_pct(self):
        """Test daily loss percentage calculation."""
        stats = TradingStats(
            start_equity=10000.0,
            daily_pnl=-500.0,
        )
        assert stats.daily_loss_pct == 5.0

    def test_record_winning_trade(self):
        """Test recording winning trade."""
        stats = TradingStats(daily_pnl=0.0)
        stats.record_trade(100.0)
        assert stats.daily_pnl == 100.0
        assert stats.consecutive_losses == 0

    def test_record_losing_trade(self):
        """Test recording losing trade."""
        stats = TradingStats(daily_pnl=0.0)
        stats.record_trade(-100.0)
        assert stats.daily_pnl == -100.0
        assert stats.consecutive_losses == 1

    def test_consecutive_losses_reset(self):
        """Test consecutive losses reset on win."""
        stats = TradingStats()
        stats.record_trade(-100.0)
        stats.record_trade(-100.0)
        assert stats.consecutive_losses == 2
        stats.record_trade(100.0)
        assert stats.consecutive_losses == 0

    def test_update_equity(self):
        """Test equity update."""
        stats = TradingStats(high_water_mark=10000.0, current_equity=10000.0)
        stats.update_equity(10500.0)
        assert stats.current_equity == 10500.0
        assert stats.high_water_mark == 10500.0

    def test_update_equity_doesnt_lower_hwm(self):
        """Test HWM doesn't decrease."""
        stats = TradingStats(high_water_mark=11000.0, current_equity=11000.0)
        stats.update_equity(10500.0)
        assert stats.current_equity == 10500.0
        assert stats.high_water_mark == 11000.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state(self):
        """Test breaker starts closed."""
        breaker = CircuitBreaker(RiskLimits())
        assert breaker.state == BreakerState.CLOSED
        assert breaker.is_closed

    def test_initialize(self):
        """Test breaker initialization."""
        breaker = CircuitBreaker(RiskLimits())
        breaker.initialize(10000.0)
        assert breaker.stats.start_equity == 10000.0
        assert breaker.stats.high_water_mark == 10000.0
        assert breaker.stats.current_equity == 10000.0

    def test_can_trade_when_closed(self):
        """Test trading allowed when closed."""
        breaker = CircuitBreaker(RiskLimits())
        breaker.initialize(10000.0)
        can_trade, reason = breaker.check_can_trade()
        assert can_trade
        assert reason is None

    def test_cannot_trade_when_open(self):
        """Test trading blocked when open."""
        breaker = CircuitBreaker(RiskLimits(max_consecutive_losses=2))
        breaker.initialize(10000.0)
        breaker.record_trade(-100.0)
        breaker.record_trade(-100.0)
        assert breaker.is_open
        can_trade, reason = breaker.check_can_trade()
        assert not can_trade
        assert "open" in reason.lower()

    def test_trip_on_max_drawdown(self):
        """Test breaker trips on max drawdown."""
        breaker = CircuitBreaker(RiskLimits(max_drawdown_pct=10.0))
        breaker.initialize(10000.0)
        breaker.update_equity(10500.0)  # New HWM
        breaker.update_equity(9000.0)  # 14.3% drawdown
        assert breaker.is_open
        assert len(breaker.trip_history) == 1
        assert breaker.trip_history[0].reason == "max_drawdown"

    def test_trip_on_max_daily_loss(self):
        """Test breaker trips on max daily loss."""
        breaker = CircuitBreaker(RiskLimits(max_daily_loss_pct=5.0))
        breaker.initialize(10000.0)
        breaker.record_trade(-600.0)  # 6% daily loss
        assert breaker.is_open
        assert breaker.trip_history[0].reason == "max_daily_loss"

    def test_trip_on_consecutive_losses(self):
        """Test breaker trips on consecutive losses."""
        breaker = CircuitBreaker(RiskLimits(max_consecutive_losses=3))
        breaker.initialize(10000.0)
        breaker.record_trade(-100.0)
        breaker.record_trade(-100.0)
        assert breaker.is_closed
        breaker.record_trade(-100.0)
        assert breaker.is_open
        assert breaker.trip_history[0].reason == "max_consecutive_losses"

    def test_trip_callback(self):
        """Test trip callback is called."""
        trips = []

        def on_trip(trip):
            trips.append(trip)

        breaker = CircuitBreaker(
            RiskLimits(max_consecutive_losses=1),
            on_trip=on_trip,
        )
        breaker.initialize(10000.0)
        breaker.record_trade(-100.0)

        assert len(trips) == 1
        assert trips[0].reason == "max_consecutive_losses"

    def test_reset(self):
        """Test breaker reset."""
        breaker = CircuitBreaker(RiskLimits(max_consecutive_losses=1))
        breaker.initialize(10000.0)
        breaker.record_trade(-100.0)
        assert breaker.is_open
        breaker.reset()
        assert breaker.is_closed

    def test_check_position_size_allowed(self):
        """Test position size within limits."""
        breaker = CircuitBreaker(RiskLimits(max_position_pct=50.0))
        breaker.initialize(10000.0)
        allowed, reason = breaker.check_position_size(4000.0)  # 40%
        assert allowed
        assert reason is None

    def test_check_position_size_blocked_pct(self):
        """Test position size blocked by percentage."""
        breaker = CircuitBreaker(RiskLimits(max_position_pct=50.0))
        breaker.initialize(10000.0)
        allowed, reason = breaker.check_position_size(6000.0)  # 60%
        assert not allowed
        assert "60.0%" in reason

    def test_check_position_size_blocked_value(self):
        """Test position size blocked by absolute value."""
        breaker = CircuitBreaker(RiskLimits(max_position_value=5000.0))
        breaker.initialize(10000.0)
        allowed, reason = breaker.check_position_size(6000.0)
        assert not allowed
        assert "6000.00" in reason

    def test_get_status(self):
        """Test status dict."""
        breaker = CircuitBreaker(RiskLimits())
        breaker.initialize(10000.0)
        status = breaker.get_status()
        assert status["state"] == "closed"
        assert status["equity"] == 10000.0
        assert "limits" in status

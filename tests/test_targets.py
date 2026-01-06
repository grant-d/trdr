"""Tests for composite scoring function.

Tests cover:
- Individual metric scaling (asymptotic and quadratic functions)
- Edge cases (zero, negative, infinity)
- Drawdown penalty
- Alpha penalty (vs buy-hold)
"""

from dataclasses import dataclass

from trdr.strategy.targets import asymptotic, quadratic, score_result


@dataclass
class MockResult:
    """Mock PaperExchangeResult for testing score_result."""

    cagr: float | None = 0.0
    calmar_ratio: float | None = 0.0
    sortino_ratio: float | None = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    trades_per_year: float = 0.0
    total_return: float = 0.0
    start_time: str | None = None
    end_time: str | None = None


class TestAsymptotic:
    """Test asymptotic scaling function."""

    def test_zero_returns_zero(self):
        assert asymptotic(0, 1.0) == 0.0

    def test_negative_returns_zero(self):
        assert asymptotic(-5, 1.0) == 0.0

    def test_large_negative_returns_zero(self):
        """Regression: negative / negative = positive, must still return 0.

        Bug: max(0, value/(value+target)) fails when |value| > target
        because -2.32 / (-2.32 + 2.0) = -2.32 / -0.32 = 7.25 (positive!)
        """
        assert asymptotic(-2.32, 2.0) == 0.0  # The actual bug case
        assert asymptotic(-100, 2.0) == 0.0  # Large negative
        assert asymptotic(-1.5, 1.0) == 0.0  # |value| > target

    def test_at_target_returns_half(self):
        assert asymptotic(1.0, 1.0) == 0.5
        assert asymptotic(1.5, 1.5) == 0.5
        assert asymptotic(100, 100) == 0.5

    def test_double_target_returns_two_thirds(self):
        # value / (value + target) = 2 / (2 + 1) = 0.667
        assert abs(asymptotic(2.0, 1.0) - 0.667) < 0.01

    def test_approaches_one_asymptotically(self):
        # Very large values approach 1.0
        assert asymptotic(1000, 1.0) > 0.99
        assert asymptotic(10000, 1.0) > 0.999

    def test_small_values(self):
        # Half of target → 1/3
        assert abs(asymptotic(0.5, 1.0) - 0.333) < 0.01


class TestQuadratic:
    """Test quadratic scaling function."""

    def test_at_target_returns_one(self):
        """Score = 1.0 at target."""
        assert quadratic(100, target=100, delta=100) == 1.0
        assert quadratic(50, target=50, delta=50) == 1.0

    def test_at_zero_returns_zero(self):
        """Score = 0 at target - delta."""
        assert quadratic(0, target=100, delta=100) == 0.0

    def test_at_double_target_returns_zero(self):
        """Score = 0 at target + delta."""
        assert quadratic(200, target=100, delta=100) == 0.0

    def test_halfway_returns_three_quarters(self):
        """Score = 0.75 at halfway points."""
        # At 50 (halfway between 0 and 100): normalized = -0.5, score = 1 - 0.25 = 0.75
        assert abs(quadratic(50, target=100, delta=100) - 0.75) < 0.01
        # At 150 (halfway between 100 and 200): normalized = 0.5, score = 0.75
        assert abs(quadratic(150, target=100, delta=100) - 0.75) < 0.01

    def test_beyond_range_clamped_to_zero(self):
        """Values beyond target ± delta clamp to 0."""
        assert quadratic(-50, target=100, delta=100) == 0.0
        assert quadratic(250, target=100, delta=100) == 0.0

    def test_symmetric_around_target(self):
        """Score is symmetric around target."""
        # 80 and 120 are equidistant from 100
        score_below = quadratic(80, target=100, delta=100)
        score_above = quadratic(120, target=100, delta=100)
        assert abs(score_below - score_above) < 0.001

    def test_different_deltas(self):
        """Different deltas change the curve width."""
        # Narrow delta = steeper falloff
        narrow = quadratic(80, target=100, delta=50)  # 20 away from 100, delta=50
        wide = quadratic(80, target=100, delta=100)  # 20 away from 100, delta=100
        assert narrow < wide  # Narrow penalizes more


class TestScoreResultBasic:
    """Test basic score_result behavior."""

    def test_good_metrics(self):
        """Good metrics produce decent score."""
        result = MockResult(
            cagr=0.30,
            calmar_ratio=3.0,
            sortino_ratio=3.0,
            profit_factor=3.0,
            win_rate=0.60,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.30,
        )
        score, _ = score_result(result)
        assert 0.4 < score < 0.8

    def test_poor_metrics_low_score(self):
        """Poor metrics produce low score."""
        result = MockResult(
            cagr=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.2,
            profit_factor=0.5,
            win_rate=0.30,
            max_drawdown=0.10,
            trades_per_year=5,
            total_return=0.0,
        )
        score, _ = score_result(result)
        assert score < 0.15

    def test_zero_trades_zero_score(self):
        """Zero trades produces zero trade score component."""
        result = MockResult(
            profit_factor=0,
            sortino_ratio=0,
            win_rate=0,
            max_drawdown=0,
            trades_per_year=0,
        )
        score, _ = score_result(result)
        assert score < 0.05


class TestDrawdownPenalty:
    """Test drawdown penalty behavior."""

    def test_low_drawdown_no_penalty(self):
        """Drawdown <= 25% should not be penalized."""
        result_5 = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.05,
            trades_per_year=70,
            total_return=0.20,
        )
        result_25 = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.25,
            trades_per_year=70,
            total_return=0.20,
        )
        score_5, details_5 = score_result(result_5)
        score_25, details_25 = score_result(result_25)
        # Both should have DD penalty 1.0
        assert "DD: 5.0% → penalty 1.00" in str(details_5)
        assert "DD: 25.0% → penalty 1.00" in str(details_25)
        assert score_5 == score_25

    def test_high_drawdown_penalized(self):
        """Drawdown > 25% should be penalized exponentially."""
        base = dict(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.50,
            trades_per_year=70,
            total_return=0.20,
        )
        score_25, _ = score_result(MockResult(**base, max_drawdown=0.25))
        score_35, _ = score_result(MockResult(**base, max_drawdown=0.35))
        score_45, _ = score_result(MockResult(**base, max_drawdown=0.45))
        # Higher DD = lower score
        assert score_35 < score_25
        assert score_45 < score_35
        # 45% DD should be heavily penalized
        assert score_45 < score_25 * 0.5


class TestAlphaPenalty:
    """Test alpha penalty (vs buy-hold) behavior."""

    def test_no_buyhold_no_penalty(self):
        """Without buy-hold return, no alpha penalty applied."""
        result = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        score, details = score_result(result, buyhold_return=None)
        assert "Alpha" not in str(details)

    def test_outperform_buyhold_no_penalty(self):
        """Beating buy-hold should not be penalized."""
        result = MockResult(
            cagr=0.50,
            calmar_ratio=3.0,
            sortino_ratio=2.0,
            profit_factor=2.5,
            win_rate=0.55,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.50,  # 50% return
        )
        score, details = score_result(result, buyhold_return=0.30)  # 30% buy-hold
        # alpha = 0.50 / 0.30 = 1.67
        assert "Alpha: 1.67x buy-hold" in str(details)
        assert "penalty 1.00" in str(details)

    def test_underperform_buyhold_penalized(self):
        """Underperforming buy-hold should be penalized."""
        result_under = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,  # 20% return
        )
        result_over = MockResult(
            cagr=0.60,
            calmar_ratio=3.0,
            sortino_ratio=2.5,
            profit_factor=3.0,
            win_rate=0.55,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.60,  # 60% return
        )
        score_under, details = score_result(result_under, buyhold_return=0.50)
        score_over, _ = score_result(result_over, buyhold_return=0.50)
        assert score_under < score_over
        # alpha = 0.20 / 0.50 = 0.40
        assert "Alpha: 0.40x buy-hold" in str(details)

    def test_alpha_floor_at_30_percent(self):
        """Alpha penalty should floor at 0.3 for normal markets."""
        result = MockResult(
            cagr=0.01,
            calmar_ratio=0.5,
            sortino_ratio=0.5,
            profit_factor=1.5,
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.01,  # 1% return
        )
        score, details = score_result(result, buyhold_return=0.20)  # Normal market
        assert "penalty 0.30" in str(details)
        assert score > 0

    def test_negative_buyhold_shows_excess_return(self):
        """Negative buyhold should show excess return, not meaningless ratio."""
        result = MockResult(
            cagr=1.05,
            calmar_ratio=16.0,
            sortino_ratio=700.0,
            profit_factor=3.3,
            win_rate=0.50,
            max_drawdown=0.065,
            trades_per_year=48,
            total_return=0.072,  # +7.2% strategy return
        )
        score, details = score_result(result, buyhold_return=-0.0155)  # -1.55% market
        details_str = str(details)
        # Should show excess return format, not ratio
        assert "vs buy-hold" in details_str
        assert "no penalty" in details_str
        # Should NOT show meaningless negative ratio
        assert "-4" not in details_str and "-5" not in details_str

    def test_negative_buyhold_no_penalty(self):
        """Beating a down market should never be penalized."""
        result = MockResult(
            cagr=0.50,
            calmar_ratio=5.0,
            sortino_ratio=3.0,
            profit_factor=2.5,
            win_rate=0.55,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.10,  # +10% in down market
        )
        score_down, _ = score_result(result, buyhold_return=-0.20)  # -20% market
        score_none, _ = score_result(result, buyhold_return=None)
        # No alpha penalty when beating down market
        assert score_down == score_none


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_infinity_sortino_handled(self):
        """Infinite Sortino (no downside deviation) should be handled."""
        result = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=float("inf"),
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        score, _ = score_result(result)
        assert 0 < score < 1

    def test_none_sortino_handled(self):
        """None Sortino should be handled as 0."""
        result = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=None,
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        score, _ = score_result(result)
        assert 0 < score < 1

    def test_infinity_profit_factor_capped(self):
        """Infinite PF (no losing trades) should be capped."""
        result = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=float("inf"),
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        score, details = score_result(result)
        # Capped at 10.0
        assert "PF: 10.00" in str(details)
        assert 0 < score < 1

    def test_win_rate_clamped(self):
        """Win rate should be clamped to [0, 1]."""
        result_over = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=1.5,  # Invalid > 1
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        result_under = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=-0.5,  # Invalid < 0
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        score_over, _ = score_result(result_over)
        score_under, _ = score_result(result_under)
        assert 0 < score_over < 1
        assert 0 < score_under < 1

    def test_zero_buyhold_return_no_penalty(self):
        """Zero buy-hold return should not cause division by zero."""
        result = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        score, details = score_result(result, buyhold_return=0.0)
        # Should skip alpha calculation
        assert "Alpha" not in str(details)


class TestDetailsOutput:
    """Test that details output is informative and correct."""

    def test_details_contains_all_metrics(self):
        """Details should contain all metric breakdowns."""
        result = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        _, details = score_result(result)
        details_str = "\n".join(details)
        assert "CAGR:" in details_str
        assert "Calmar:" in details_str
        assert "Sortino:" in details_str
        assert "PF:" in details_str
        assert "WR:" in details_str
        assert "Trades:" in details_str
        assert "DD:" in details_str
        assert "Score:" in details_str

    def test_details_shows_alpha_when_provided(self):
        """Details should show alpha when buy-hold return provided."""
        result = MockResult(
            cagr=0.20,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.50,
            max_drawdown=0.10,
            trades_per_year=70,
            total_return=0.20,
        )
        _, details = score_result(result, buyhold_return=0.30)
        details_str = "\n".join(details)
        assert "Alpha:" in details_str
        assert "buy-hold" in details_str

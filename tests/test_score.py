"""Tests for composite scoring function.

Tests cover:
- Individual metric scaling (asymptotic and quadratic functions)
- Edge cases (zero, negative, infinity)
- Drawdown penalty
- Alpha penalty (vs buy-hold)
- Permutations showing score behavior across metric ranges
"""

import pytest

from trdr.strategy.score import asymptotic, quadratic, compute_composite_score


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
        assert asymptotic(-100, 2.0) == 0.0   # Large negative
        assert asymptotic(-1.5, 1.0) == 0.0   # |value| > target

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
        wide = quadratic(80, target=100, delta=100)   # 20 away from 100, delta=100
        assert narrow < wide  # Narrow penalizes more


class TestCompositeScoreBasic:
    """Test basic composite score behavior.

    Note: Without bars, CAGR and Calmar are 0, which is 60% of the score.
    These tests verify the secondary metrics (40%) behave correctly.
    """

    def test_good_secondary_metrics(self):
        """Good secondary metrics should produce decent score (max 40% without bars)."""
        score, _ = compute_composite_score(
            profit_factor=5.0,
            sortino=3.0,
            pnl=10000,
            win_rate=0.65,
            max_drawdown=0.05,
            total_trades=50,
        )
        # Without bars: CAGR=0, Calmar=0 (60% lost)
        # Secondary metrics max out around 0.35-0.40
        assert 0.2 < score < 0.5

    def test_poor_metrics_low_score(self):
        """Poor metrics should produce low score."""
        score, _ = compute_composite_score(
            profit_factor=0.5,
            sortino=0.2,
            pnl=100,
            win_rate=0.30,
            max_drawdown=0.10,
            total_trades=5,
        )
        assert score < 0.2

    def test_zero_trades_near_zero_score(self):
        """Zero trades should produce very low score."""
        score, _ = compute_composite_score(
            profit_factor=0,
            sortino=0,
            pnl=0,
            win_rate=0,
            max_drawdown=0,
            total_trades=0,
        )
        assert score < 0.01


class TestDrawdownPenalty:
    """Test drawdown penalty behavior."""

    def test_low_drawdown_no_penalty(self):
        """Drawdown <= 25% should not be penalized."""
        score_5pct, details_5 = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.05,
            total_trades=20,
        )
        score_25pct, details_25 = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.25,
            total_trades=20,
        )
        # Both should have DD penalty 1.0
        assert "DD: 5.0% → penalty 1.00" in str(details_5)
        assert "DD: 25.0% → penalty 1.00" in str(details_25)
        assert score_5pct == score_25pct

    def test_high_drawdown_penalized(self):
        """Drawdown > 25% should be penalized exponentially."""
        score_25pct, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.25,
            total_trades=20,
        )
        score_35pct, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.35,
            total_trades=20,
        )
        score_45pct, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.45,
            total_trades=20,
        )
        # Higher DD = lower score
        assert score_35pct < score_25pct
        assert score_45pct < score_35pct
        # 45% DD should be heavily penalized
        assert score_45pct < score_25pct * 0.5


class TestAlphaPenalty:
    """Test alpha penalty (vs buy-hold) behavior."""

    def test_no_buyhold_no_penalty(self):
        """Without buy-hold return, no alpha penalty applied."""
        score, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            buyhold_return=None,
        )
        assert "Alpha" not in str(details)

    def test_outperform_buyhold_no_penalty(self):
        """Beating buy-hold should not be penalized."""
        # Strategy: 50% return, Buy-hold: 30% return → alpha = 1.67x
        score, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=5000,  # 50% return on 10k
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.30,  # 30%
        )
        assert "Alpha: 1.67x" in str(details)
        assert "penalty 1.00" in str(details)

    def test_underperform_buyhold_penalized(self):
        """Underperforming buy-hold should be penalized."""
        # Strategy: 20% return, Buy-hold: 50% return → alpha = 0.4x
        score_under, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,  # 20% return on 10k
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.50,  # 50%
        )
        # Same metrics but beating buy-hold
        score_over, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=6000,  # 60% return on 10k
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.50,  # 50%
        )
        assert score_under < score_over
        assert "Alpha: 0.40x" in str(details)

    def test_alpha_penalty_proportional(self):
        """Alpha penalty should be proportional to underperformance."""
        # Normal market (buyhold < 25%): penalty = alpha directly
        # 80% of buy-hold → 0.8 penalty
        score_80pct, details_80 = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=1600,  # 16% return
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.20,  # 20% (normal market)
        )
        # 50% of buy-hold → 0.5 penalty
        score_50pct, details_50 = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=1000,  # 10% return
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.20,  # 20% (normal market)
        )
        assert "penalty 0.80" in str(details_80)
        assert "penalty 0.50" in str(details_50)
        assert score_80pct > score_50pct

    def test_alpha_floor_at_30_percent(self):
        """Alpha penalty should floor at 0.3 (not 0) for normal markets."""
        score, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=100,  # 1% return
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.20,  # 20% (normal market, not mega-trend)
        )
        assert "penalty 0.30" in str(details)
        assert score > 0  # Should not be zero


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_infinity_sortino_handled(self):
        """Infinite Sortino (no downside deviation) should be handled."""
        score, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=float("inf"),
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
        )
        assert 0 < score < 1

    def test_none_sortino_handled(self):
        """None Sortino should be handled as 0."""
        score, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=None,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
        )
        assert 0 < score < 1

    def test_infinity_profit_factor_capped(self):
        """Infinite PF (no losing trades) should be capped."""
        score, details = compute_composite_score(
            profit_factor=float("inf"),
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
        )
        # Capped at 10.0
        assert "PF: 10.00" in str(details)
        assert 0 < score < 1

    def test_negative_pnl_gives_zero_cagr(self):
        """Negative P&L without bars gives CAGR 0."""
        score, details = compute_composite_score(
            profit_factor=0.5,
            sortino=0.5,
            pnl=-5000,
            win_rate=0.40,
            max_drawdown=0.25,
            total_trades=20,
        )
        # Without bars, CAGR = 0 (no period info)
        assert "CAGR: 0.0%" in str(details)

    def test_win_rate_clamped(self):
        """Win rate should be clamped to [0, 1]."""
        score_over, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=1.5,  # Invalid > 1
            max_drawdown=0.10,
            total_trades=20,
        )
        score_under, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=-0.5,  # Invalid < 0
            max_drawdown=0.10,
            total_trades=20,
        )
        assert 0 < score_over < 1
        assert 0 < score_under < 1

    def test_zero_buyhold_return_no_penalty(self):
        """Zero buy-hold return should not cause division by zero."""
        score, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.0,
        )
        # Should skip alpha calculation
        assert "Alpha" not in str(details)


class TestScoreWithoutBars:
    """Test scoring behavior without bar data (CAGR/Calmar = 0)."""

    def test_secondary_metrics_only(self):
        """Without bars, only secondary metrics (40%) contribute."""
        score, details = compute_composite_score(
            profit_factor=2.0,
            sortino=2.0,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=50,  # 50 trades, no period = uses raw value
        )
        # CAGR and Calmar are 0 (60%), so max score is ~0.40
        assert 0.15 < score < 0.40
        assert "CAGR: 0.0%" in str(details)
        assert "Calmar: 0.00" in str(details)


class TestDetailsOutput:
    """Test that details output is informative and correct."""

    def test_details_contains_all_metrics(self):
        """Details should contain all metric breakdowns."""
        _, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
        )
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
        _, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.30,
        )
        details_str = "\n".join(details)
        assert "Alpha:" in details_str
        assert "buy-hold" in details_str

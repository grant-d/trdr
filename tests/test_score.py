"""Tests for composite scoring function.

Tests cover:
- Individual metric scaling (asymptotic function)
- Edge cases (zero, negative, infinity)
- Drawdown penalty
- Alpha penalty (vs buy-hold)
- Permutations showing score behavior across metric ranges
"""

import pytest

from trdr.strategy.score import asymptotic, compute_composite_score


class TestAsymptotic:
    """Test asymptotic scaling function."""

    def test_zero_returns_zero(self):
        assert asymptotic(0, 1.0) == 0.0

    def test_negative_returns_zero(self):
        assert asymptotic(-5, 1.0) == 0.0

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


class TestCompositeScoreBasic:
    """Test basic composite score behavior."""

    def test_perfect_metrics_high_score(self):
        """Excellent metrics should produce high score."""
        score, _ = compute_composite_score(
            profit_factor=5.0,
            sortino=3.0,
            pnl=10000,
            win_rate=0.65,
            max_drawdown=0.05,
            total_trades=50,
        )
        assert score > 0.7

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
        assert score < 0.3

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
        """Drawdown <= 20% should not be penalized."""
        score_5pct, details_5 = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.05,
            total_trades=20,
        )
        score_20pct, details_20 = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.20,
            total_trades=20,
        )
        # Both should have penalty 1.0
        assert "penalty 1.00" in str(details_5)
        assert "penalty 1.00" in str(details_20)
        assert score_5pct == score_20pct

    def test_high_drawdown_penalized(self):
        """Drawdown > 20% should be penalized exponentially."""
        score_20pct, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.20,
            total_trades=20,
        )
        score_30pct, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.30,
            total_trades=20,
        )
        score_40pct, _ = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2000,
            win_rate=0.50,
            max_drawdown=0.40,
            total_trades=20,
        )
        # Higher DD = lower score
        assert score_30pct < score_20pct
        assert score_40pct < score_30pct
        # 40% DD should be heavily penalized
        assert score_40pct < score_20pct * 0.5


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
        # 80% of buy-hold → 0.8 penalty
        score_80pct, details_80 = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=4000,  # 40% return
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.50,  # 50%
        )
        # 50% of buy-hold → 0.5 penalty
        score_50pct, details_50 = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=2500,  # 25% return
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=0.50,  # 50%
        )
        assert "penalty 0.80" in str(details_80)
        assert "penalty 0.50" in str(details_50)
        assert score_80pct > score_50pct

    def test_alpha_floor_at_10_percent(self):
        """Alpha penalty should floor at 0.1 (not 0)."""
        score, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=100,  # 1% return
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=10000,
            buyhold_return=1.00,  # 100%
        )
        assert "penalty 0.10" in str(details)
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

    def test_negative_pnl_treated_as_zero(self):
        """Negative P&L should be treated as 0 for scoring."""
        score, details = compute_composite_score(
            profit_factor=0.5,
            sortino=0.5,
            pnl=-5000,
            win_rate=0.40,
            max_drawdown=0.25,
            total_trades=20,
        )
        assert "P&L: $0" in str(details)

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


class TestScorePermutations:
    """Permutations showing score behavior across metric ranges.

    These tests document expected scores for various metric combinations
    to help understand the scoring function's behavior.
    """

    @pytest.mark.parametrize(
        "pf,sortino,pnl,wr,trades,dd,expected_min,expected_max,description",
        [
            # Baseline scenarios
            (1.5, 1.0, 1000, 0.40, 10, 0.10, 0.45, 0.55, "All at target (score ~0.5)"),
            (3.0, 2.0, 5000, 0.60, 30, 0.10, 0.65, 0.80, "Good metrics"),
            (5.0, 3.0, 10000, 0.70, 50, 0.05, 0.75, 0.90, "Excellent metrics"),

            # Single metric variations (geometric mean pulls down quickly)
            (0.5, 1.0, 1000, 0.40, 10, 0.10, 0.35, 0.50, "Poor PF only"),
            (1.5, 0.2, 1000, 0.40, 10, 0.10, 0.30, 0.45, "Poor Sortino only"),
            (1.5, 1.0, 100, 0.40, 10, 0.10, 0.30, 0.45, "Poor P&L only"),
            (1.5, 1.0, 1000, 0.20, 10, 0.10, 0.35, 0.50, "Poor WR only"),
            (1.5, 1.0, 1000, 0.40, 2, 0.10, 0.30, 0.45, "Few trades only"),

            # Drawdown impact
            (2.0, 1.5, 2000, 0.50, 20, 0.15, 0.50, 0.65, "DD 15% (no penalty)"),
            (2.0, 1.5, 2000, 0.50, 20, 0.25, 0.30, 0.45, "DD 25% (penalized)"),
            (2.0, 1.5, 2000, 0.50, 20, 0.35, 0.10, 0.25, "DD 35% (heavy penalty)"),

            # Extreme cases
            (10.0, 5.0, 50000, 0.80, 100, 0.05, 0.80, 0.95, "Exceptional metrics"),
            (0.8, 0.5, 500, 0.35, 5, 0.30, 0.05, 0.15, "Poor across the board"),
        ],
    )
    def test_score_ranges(
        self, pf, sortino, pnl, wr, trades, dd, expected_min, expected_max, description
    ):
        """Verify scores fall within expected ranges for various metric combinations."""
        score, _ = compute_composite_score(
            profit_factor=pf,
            sortino=sortino,
            pnl=pnl,
            win_rate=wr,
            max_drawdown=dd,
            total_trades=trades,
        )
        assert expected_min <= score <= expected_max, (
            f"{description}: score {score:.3f} not in [{expected_min}, {expected_max}]"
        )

    @pytest.mark.parametrize(
        "strategy_return_pct,buyhold_return_pct,expected_penalty_min,expected_penalty_max",
        [
            (150, 100, 1.0, 1.0),    # 1.5x buy-hold → no penalty
            (100, 100, 1.0, 1.0),    # 1.0x buy-hold → no penalty
            (80, 100, 0.75, 0.85),   # 0.8x buy-hold → ~0.8 penalty
            (50, 100, 0.45, 0.55),   # 0.5x buy-hold → ~0.5 penalty
            (20, 100, 0.15, 0.25),   # 0.2x buy-hold → ~0.2 penalty
            (5, 100, 0.10, 0.10),    # 0.05x buy-hold → floor at 0.1
        ],
    )
    def test_alpha_penalty_ranges(
        self, strategy_return_pct, buyhold_return_pct, expected_penalty_min, expected_penalty_max
    ):
        """Verify alpha penalty scales correctly with underperformance."""
        initial_capital = 10000
        pnl = initial_capital * (strategy_return_pct / 100)
        buyhold_return = buyhold_return_pct / 100

        _, details = compute_composite_score(
            profit_factor=2.0,
            sortino=1.5,
            pnl=pnl,
            win_rate=0.50,
            max_drawdown=0.10,
            total_trades=20,
            initial_capital=initial_capital,
            buyhold_return=buyhold_return,
        )

        # Extract penalty from details
        alpha_line = [d for d in details if "Alpha" in d][0]
        penalty = float(alpha_line.split("penalty ")[1])

        assert expected_penalty_min <= penalty <= expected_penalty_max, (
            f"Strategy {strategy_return_pct}% vs buy-hold {buyhold_return_pct}%: "
            f"penalty {penalty:.2f} not in [{expected_penalty_min}, {expected_penalty_max}]"
        )


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
        assert "PF:" in details_str
        assert "Sortino:" in details_str
        assert "P&L:" in details_str
        assert "WR:" in details_str
        assert "Trades:" in details_str
        assert "DD:" in details_str
        assert "Composite:" in details_str

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

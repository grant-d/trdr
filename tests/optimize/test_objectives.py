"""Tests for objective functions."""

import pytest

from trdr.optimize.objectives import (
    OBJECTIVES_CORE,
    OBJECTIVES_EXTENDED,
    ObjectiveResult,
    calculate_objectives,
)


class TestObjectiveResult:
    def test_to_minimization_negates_maximize_objectives(self):
        result = ObjectiveResult(
            sharpe=2.0,
            max_drawdown=0.1,
            win_rate=0.6,
            profit_factor=2.5,
            sortino=1.5,
            calmar=3.0,
            total_trades=50,
        )

        # Sharpe is maximize -> negated
        minimized = result.to_minimization(["sharpe"])
        assert minimized == [-2.0]

        # max_drawdown is minimize -> stays positive
        minimized = result.to_minimization(["max_drawdown"])
        assert minimized == [0.1]

    def test_to_minimization_handles_none_values(self):
        result = ObjectiveResult(
            sharpe=1.0,
            max_drawdown=0.15,
            win_rate=0.5,
            profit_factor=1.5,
            sortino=None,
            calmar=None,
            total_trades=10,
        )

        minimized = result.to_minimization(["sortino", "calmar"])
        assert minimized == [0.0, 0.0]  # None -> 0.0, then negated

    def test_to_minimization_caps_profit_factor(self):
        result = ObjectiveResult(
            sharpe=1.0,
            max_drawdown=0.1,
            win_rate=0.9,
            profit_factor=100.0,  # Very high
            total_trades=5,
        )

        minimized = result.to_minimization(["profit_factor"])
        assert minimized == [-10.0]  # Capped at 10, then negated

    def test_unknown_objective_raises(self):
        result = ObjectiveResult(
            sharpe=1.0,
            max_drawdown=0.1,
            win_rate=0.5,
            profit_factor=1.5,
        )

        with pytest.raises(ValueError, match="Unknown objective"):
            result.to_minimization(["unknown_metric"])


class TestCalculateObjectives:
    def test_handles_inf_values(self):
        """calculate_objectives should cap inf values for stability."""

        class MockResult:
            sharpe_ratio = float("inf")
            sortino_ratio = float("inf")
            calmar_ratio = float("inf")
            profit_factor = float("inf")
            cagr = float("inf")
            max_drawdown = 0.05
            win_rate = 0.8
            total_trades = 20
            total_return = 2.0

        result = calculate_objectives(MockResult(), buyhold_return=0.5)

        assert result.sharpe == 5.0  # Capped
        assert result.sortino == 5.0  # Capped
        assert result.calmar == 5.0  # Capped
        assert result.profit_factor == 10.0  # Capped
        assert result.cagr == 10.0  # Capped
        assert result.alpha == 4.0  # 2.0 / 0.5

    def test_handles_none_ratios(self):
        """calculate_objectives should handle None for ratios."""

        class MockResult:
            sharpe_ratio = None
            sortino_ratio = None
            calmar_ratio = None
            profit_factor = 1.5
            cagr = None
            max_drawdown = 0.1
            win_rate = 0.5
            total_trades = 5
            total_return = 0.2

        result = calculate_objectives(MockResult())

        assert result.sharpe == 0.0
        assert result.sortino is None
        assert result.calmar is None
        assert result.cagr is None
        assert result.alpha is None  # No buyhold_return provided


class TestObjectiveConstants:
    def test_core_objectives(self):
        assert "sharpe" in OBJECTIVES_CORE
        assert "max_drawdown" in OBJECTIVES_CORE
        assert "profit_factor" in OBJECTIVES_CORE
        assert len(OBJECTIVES_CORE) == 3

    def test_extended_objectives(self):
        assert "win_rate" in OBJECTIVES_EXTENDED
        assert len(OBJECTIVES_EXTENDED) == 4

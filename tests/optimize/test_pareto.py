"""Tests for Pareto selection interface."""

import numpy as np
import pytest

from trdr.optimize.multi_objective import MooResult
from trdr.optimize.pareto import (
    filter_pareto_front,
    rank_pareto_solutions,
)


@pytest.fixture
def mock_moo_result():
    """Create a mock MooResult with 3 solutions."""
    return MooResult(
        pareto_params=np.array([
            [10.0, 0.5],   # Solution 0
            [15.0, 0.8],   # Solution 1
            [20.0, 0.3],   # Solution 2
        ]),
        pareto_objectives=np.array([
            [-2.0, 0.10, -0.60],   # sharpe=2.0, max_dd=0.10, win_rate=0.60
            [-1.5, 0.08, -0.70],   # sharpe=1.5, max_dd=0.08, win_rate=0.70
            [-2.5, 0.15, -0.55],   # sharpe=2.5, max_dd=0.15, win_rate=0.55
        ]),
        param_names=["lookback", "threshold"],
        param_dtypes=["int", "float"],
        objective_names=["sharpe", "max_drawdown", "win_rate"],
        n_generations=50,
        n_evaluations=2500,
    )


class TestMooResult:
    def test_n_solutions(self, mock_moo_result):
        assert mock_moo_result.n_solutions == 3

    def test_get_params_dict(self, mock_moo_result):
        params = mock_moo_result.get_params_dict(0)
        assert params == {"lookback": 10, "threshold": 0.5}  # lookback is int

    def test_get_objectives_dict_undoes_minimization(self, mock_moo_result):
        obj = mock_moo_result.get_objectives_dict(0)

        # sharpe was negated for minimization, should be positive now
        assert obj["sharpe"] == 2.0

        # max_drawdown stays as-is
        assert obj["max_drawdown"] == 0.10

        # win_rate was negated, should be positive
        assert obj["win_rate"] == 0.60


class TestFilterParetoFront:
    def test_filter_by_single_constraint(self, mock_moo_result):
        # Only solutions with max_drawdown <= 12%
        valid = filter_pareto_front(
            mock_moo_result,
            {"max_drawdown": (None, 0.12)},
        )
        assert valid == [0, 1]  # Solution 2 has 0.15

    def test_filter_by_multiple_constraints(self, mock_moo_result):
        # max_drawdown <= 15% AND sharpe >= 2.0
        valid = filter_pareto_front(
            mock_moo_result,
            {
                "max_drawdown": (None, 0.15),
                "sharpe": (2.0, None),
            },
        )
        assert valid == [0, 2]  # Solution 1 has sharpe=1.5

    def test_filter_returns_empty_when_no_match(self, mock_moo_result):
        valid = filter_pareto_front(
            mock_moo_result,
            {"sharpe": (10.0, None)},  # No solution has sharpe >= 10
        )
        assert valid == []


class TestRankParetoSolutions:
    def test_rank_by_default_weights(self, mock_moo_result):
        ranked = rank_pareto_solutions(mock_moo_result)

        # Default weights favor sharpe (1.0), profit_factor (0.5), penalize drawdown (-1.0)
        # Since we don't have profit_factor in mock, just sharpe - drawdown
        # Solution 2: 2.5 - 0.15 = 2.35
        # Solution 0: 2.0 - 0.10 = 1.90
        # Solution 1: 1.5 - 0.08 = 1.42

        assert ranked[0][0] == 2  # Solution 2 is best
        assert ranked[1][0] == 0  # Solution 0 is second
        assert ranked[2][0] == 1  # Solution 1 is third

    def test_rank_by_custom_weights(self, mock_moo_result):
        # Favor low drawdown only
        ranked = rank_pareto_solutions(
            mock_moo_result,
            weights={"max_drawdown": -10.0},  # Heavily penalize drawdown
        )

        # Solution 1 has lowest drawdown (0.08)
        assert ranked[0][0] == 1

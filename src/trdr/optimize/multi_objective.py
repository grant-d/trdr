"""Multi-objective optimization using pymoo.

Integrates NSGA-II/III with strategy backtesting for Pareto-optimal parameter sets.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .objectives import OBJECTIVES_CORE, calculate_objectives

if TYPE_CHECKING:
    from ..backtest.types import PaperExchangeConfig, PaperExchangeResult
    from ..data import Bar
    from ..strategy import BaseStrategy


@dataclass
class ParamBounds:
    """Parameter bounds for optimization.

    Args:
        name: Parameter name
        lower: Lower bound
        upper: Upper bound
        dtype: "float" or "int"
    """

    name: str
    lower: float
    upper: float
    dtype: str = "float"


@dataclass
class MooConfig:
    """Configuration for multi-objective optimization.

    Args:
        param_bounds: List of parameter bounds
        objectives: List of objective names (from objectives.py)
        population_size: GA population size
        generations: Number of generations to run
        min_trades: Minimum trades for valid solution (constraint)
        seed: Random seed for reproducibility
    """

    param_bounds: list[ParamBounds]
    objectives: list[str] = field(default_factory=lambda: OBJECTIVES_CORE.copy())
    population_size: int = 50
    generations: int = 100
    min_trades: int = 10
    seed: int | None = None


@dataclass
class MooResult:
    """Result from multi-objective optimization.

    Args:
        pareto_params: Parameter values on Pareto front (N x n_var)
        pareto_objectives: Objective values on Pareto front (N x n_obj)
        param_names: Names of parameters
        param_dtypes: Dtypes for parameters ("float" or "int")
        objective_names: Names of objectives
        n_generations: Actual generations run
        n_evaluations: Total fitness evaluations
    """

    pareto_params: np.ndarray
    pareto_objectives: np.ndarray
    param_names: list[str]
    param_dtypes: list[str]
    objective_names: list[str]
    n_generations: int
    n_evaluations: int
    period_days: float | None = None
    buyhold_return: float | None = None

    @property
    def n_solutions(self) -> int:
        """Number of Pareto-optimal solutions."""
        return len(self.pareto_params)

    def get_params_dict(self, index: int) -> dict[str, float | int]:
        """Get parameter dictionary for solution at index."""
        result = {}
        for i, name in enumerate(self.param_names):
            val = self.pareto_params[index][i]
            if self.param_dtypes[i] == "int":
                result[name] = int(round(val))
            else:
                result[name] = float(val)
        return result

    def get_objectives_dict(self, index: int) -> dict[str, float]:
        """Get objective dictionary for solution at index.

        Values are converted back to natural form (not minimization).
        """
        raw = self.pareto_objectives[index]
        result = {}
        # Objectives that were negated for minimization (maximize objectives)
        negate_objectives = {
            "sharpe",
            "win_rate",
            "profit_factor",
            "sortino",
            "calmar",
            "cagr",
            "alpha",
            "total_trades",
        }
        for i, name in enumerate(self.objective_names):
            if name in negate_objectives:
                result[name] = -raw[i]  # Undo negation
            else:
                result[name] = raw[i]  # max_drawdown stays as-is
        return result


class StrategyOptimizationProblem(ElementwiseProblem):
    """pymoo Problem for strategy parameter optimization.

    Evaluates strategy parameters by running backtests and computing objectives.
    """

    def __init__(
        self,
        strategy_factory: Callable[[dict], "BaseStrategy"],
        bars: dict[str, list["Bar"]],
        exchange_config: "PaperExchangeConfig",
        moo_config: MooConfig,
    ):
        """Initialize optimization problem.

        Args:
            strategy_factory: Callable that creates strategy from param dict
            bars: Bar data keyed by "symbol:timeframe"
            exchange_config: PaperExchange configuration
            moo_config: Multi-objective optimization configuration
        """
        self.strategy_factory = strategy_factory
        self.bars = bars
        self.exchange_config = exchange_config
        self.moo_config = moo_config
        self.param_names = [p.name for p in moo_config.param_bounds]
        self._error_count = 0

        # Calculate buy-hold return for alpha
        self.buyhold_return = self._calculate_buyhold_return()

        # Build bounds arrays
        xl = np.array([p.lower for p in moo_config.param_bounds])
        xu = np.array([p.upper for p in moo_config.param_bounds])

        # Constraint: minimum trades
        n_ieq_constr = 1 if moo_config.min_trades > 0 else 0

        super().__init__(
            n_var=len(moo_config.param_bounds),
            n_obj=len(moo_config.objectives),
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
        )

    def _calculate_buyhold_return(self) -> float:
        """Calculate buy-and-hold return from bars."""
        # Get primary feed if present; fall back to first bar list
        primary_key = str(self.exchange_config.primary_feed)
        primary_bars = self.bars.get(primary_key) or next(iter(self.bars.values()))
        if len(primary_bars) < 2:
            return 0.0
        return (primary_bars[-1].close / primary_bars[0].close) - 1

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate single parameter set.

        Args:
            x: Parameter values array
            out: Output dict for objectives ("F") and constraints ("G")
        """
        # Convert to parameter dict, respecting dtypes
        params = {}
        for i, bound in enumerate(self.moo_config.param_bounds):
            if bound.dtype == "int":
                params[bound.name] = int(round(x[i]))
            else:
                params[bound.name] = float(x[i])

        # Run backtest
        try:
            result = self._run_backtest(params)
            objectives = calculate_objectives(result, self.buyhold_return)

            # Get objectives in minimization form
            out["F"] = objectives.to_minimization(self.moo_config.objectives)

            # Constraint: min_trades - total_trades <= 0 (satisfied when trades >= min)
            if self.moo_config.min_trades > 0:
                out["G"] = [self.moo_config.min_trades - objectives.total_trades]

        except Exception as exc:
            self._error_count += 1
            if self._error_count <= 3:
                print(f"MOO backtest failed for params {params}: {exc}")
            # Failed backtest: return worst-case objectives
            out["F"] = [1e6] * len(self.moo_config.objectives)
            if self.moo_config.min_trades > 0:
                out["G"] = [1e6]  # Constraint violated

    def _run_backtest(self, params: dict) -> "PaperExchangeResult":
        """Run backtest with given parameters."""
        from ..backtest import PaperExchange

        strategy = self.strategy_factory(params)
        engine = PaperExchange(self.exchange_config, strategy)
        return engine.run(self.bars)


def run_moo(
    strategy_factory: Callable[[dict], "BaseStrategy"],
    bars: dict[str, list["Bar"]],
    exchange_config: "PaperExchangeConfig",
    moo_config: MooConfig,
    verbose: bool = True,
) -> MooResult:
    """Run multi-objective optimization.

    Uses NSGA-II for 2-3 objectives, NSGA-III for 4+ objectives.

    Args:
        strategy_factory: Callable that creates strategy from param dict
        bars: Bar data keyed by "symbol:timeframe"
        exchange_config: PaperExchange configuration
        moo_config: Optimization configuration
        verbose: Print progress

    Returns:
        MOOResult with Pareto-optimal solutions
    """
    problem = StrategyOptimizationProblem(
        strategy_factory=strategy_factory,
        bars=bars,
        exchange_config=exchange_config,
        moo_config=moo_config,
    )

    # Select algorithm based on number of objectives
    n_obj = len(moo_config.objectives)
    if n_obj <= 3:
        algorithm = NSGA2(pop_size=moo_config.population_size)
    else:
        # NSGA-III for many objectives - requires reference directions
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        algorithm = NSGA3(pop_size=moo_config.population_size, ref_dirs=ref_dirs)

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ("n_gen", moo_config.generations),
        seed=moo_config.seed,
        verbose=verbose,
    )

    primary_key = str(exchange_config.primary_feed)
    primary_bars = bars.get(primary_key) or next(iter(bars.values()))
    period_days = 0.0
    if len(primary_bars) >= 2:
        start = datetime.fromisoformat(primary_bars[0].timestamp.replace("Z", "+00:00"))
        end = datetime.fromisoformat(primary_bars[-1].timestamp.replace("Z", "+00:00"))
        period_days = max(0.0, (end - start).total_seconds() / 86400.0)

    return MooResult(
        pareto_params=res.X if res.X is not None else np.array([]),
        pareto_objectives=res.F if res.F is not None else np.array([]),
        param_names=[p.name for p in moo_config.param_bounds],
        param_dtypes=[p.dtype for p in moo_config.param_bounds],
        objective_names=moo_config.objectives,
        n_generations=res.algorithm.n_gen if res.algorithm else 0,
        n_evaluations=res.algorithm.evaluator.n_eval if res.algorithm else 0,
        period_days=period_days,
        buyhold_return=problem.buyhold_return,
    )

"""Objective functions for multi-objective optimization.

Objectives are extracted from PaperExchangeResult and formatted for pymoo.
All objectives are converted to minimization form (pymoo convention).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backtest.types import PaperExchangeResult


@dataclass
class ObjectiveResult:
    """Computed objectives from a backtest result.

    All values are in their natural form (higher is better for Sharpe, etc.).
    Use to_minimization() to convert for pymoo.
    """

    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sortino: float | None = None
    calmar: float | None = None
    cagr: float | None = None
    alpha: float | None = None  # Strategy return / buy-hold return
    total_trades: int = 0

    def to_minimization(self, objectives: list[str]) -> list[float]:
        """Convert objectives to minimization form for pymoo.

        Args:
            objectives: List of objective names to extract

        Returns:
            List of objective values in minimization form
        """
        result = []
        for obj in objectives:
            value = self._get_objective(obj)
            result.append(value)
        return result

    def _get_objective(self, name: str) -> float:
        """Get objective value in minimization form.

        Maximization objectives are negated.
        """
        if name == "sharpe":
            return -self.sharpe  # Maximize -> minimize negative
        elif name == "max_drawdown":
            return self.max_drawdown  # Minimize as-is
        elif name == "win_rate":
            return -self.win_rate  # Maximize -> minimize negative
        elif name == "profit_factor":
            return -min(self.profit_factor, 10.0)  # Cap at 10, negate
        elif name == "sortino":
            val = self.sortino if self.sortino is not None else 0.0
            return -val  # Maximize -> minimize negative
        elif name == "calmar":
            val = self.calmar if self.calmar is not None else 0.0
            return -val  # Maximize -> minimize negative
        elif name == "cagr":
            val = self.cagr if self.cagr is not None else 0.0
            return -val  # Maximize -> minimize negative
        elif name == "alpha":
            val = self.alpha if self.alpha is not None else 0.0
            return -val  # Maximize -> minimize negative
        elif name == "total_trades":
            return -self.total_trades  # More trades often better, negate
        else:
            raise ValueError(f"Unknown objective: {name}")


def calculate_objectives(
    result: "PaperExchangeResult",
    buyhold_return: float | None = None,
) -> ObjectiveResult:
    """Calculate all objectives from backtest result.

    Args:
        result: PaperExchangeResult from backtesting
        buyhold_return: Buy-and-hold return for alpha calculation (optional)

    Returns:
        ObjectiveResult with all computed metrics
    """
    # Handle infinite profit factor
    pf = result.profit_factor
    if pf == float("inf"):
        pf = 10.0  # Cap for optimization stability

    # Handle None ratios (too few trades)
    sharpe = result.sharpe_ratio if result.sharpe_ratio is not None else 0.0
    if sharpe == float("inf"):
        sharpe = 5.0  # Cap

    sortino = result.sortino_ratio
    if sortino == float("inf"):
        sortino = 5.0

    calmar = result.calmar_ratio
    if calmar == float("inf"):
        calmar = 5.0

    cagr = result.cagr
    if cagr == float("inf"):
        cagr = 10.0  # Cap at 1000%

    # Alpha: strategy return / buy-hold return (or excess return in down markets)
    alpha = None
    if buyhold_return is not None:
        strategy_return = result.total_return
        if buyhold_return > 0:
            alpha = strategy_return / buyhold_return if strategy_return else 0.0
        else:
            alpha = strategy_return - buyhold_return
        if alpha == float("inf"):
            alpha = 10.0  # Cap
        elif alpha == float("-inf"):
            alpha = -10.0  # Cap

    return ObjectiveResult(
        sharpe=sharpe,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        profit_factor=pf,
        sortino=sortino,
        calmar=calmar,
        cagr=cagr,
        alpha=alpha,
        total_trades=result.total_trades,
    )


# Standard objective sets
OBJECTIVES_CORE = ["sharpe", "max_drawdown", "profit_factor"]
OBJECTIVES_EXTENDED = ["sharpe", "max_drawdown", "win_rate", "profit_factor"]
OBJECTIVES_FULL = [
    "cagr", "calmar", "sortino", "profit_factor", "win_rate", "total_trades", "max_drawdown", "alpha"
]

# Objective descriptions for display
OBJECTIVE_DESCRIPTIONS = {
    "sharpe": ("Sharpe Ratio", "Maximize", "Risk-adjusted return"),
    "max_drawdown": ("Max Drawdown", "Minimize", "Peak-to-trough decline"),
    "win_rate": ("Win Rate", "Maximize", "Fraction of winning trades"),
    "profit_factor": ("Profit Factor", "Maximize", "Gross profit / gross loss"),
    "sortino": ("Sortino Ratio", "Maximize", "Return / downside risk"),
    "calmar": ("Calmar Ratio", "Maximize", "CAGR / max drawdown"),
    "cagr": ("CAGR", "Maximize", "Compound annual growth rate"),
    "alpha": ("Alpha", "Maximize", "Strategy return / buy-hold return"),
    "total_trades": ("Total Trades", "Maximize", "Number of trades"),
}

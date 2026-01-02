"""Run archival for self-improvement analysis."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TradeRecord:
    """Single trade record."""

    timestamp: str
    action: str  # "buy" or "sell"
    symbol: str
    price: float
    qty: float
    reason: str
    pnl: float | None = None  # Only for closing trades
    poc: float | None = None  # POC at time of trade
    va_high: float | None = None
    va_low: float | None = None


@dataclass
class RunMetrics:
    """Aggregate metrics for a run."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float | None = None  # Requires more calculation


@dataclass
class RunRecord:
    """Complete record of a trading run."""

    run_id: str
    start_time: str
    end_time: str | None
    mode: str  # "paper" or "live"
    symbol: str
    config: dict[str, Any]  # Strategy parameters
    metrics: RunMetrics | None
    trades: list[TradeRecord]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "mode": self.mode,
            "symbol": self.symbol,
            "config": self.config,
            "metrics": asdict(self.metrics) if self.metrics else None,
            "trades": [asdict(t) for t in self.trades],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        """Create RunRecord from dictionary."""
        metrics = None
        if data.get("metrics"):
            metrics = RunMetrics(**data["metrics"])

        trades = [TradeRecord(**t) for t in data.get("trades", [])]

        return cls(
            run_id=data["run_id"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            mode=data["mode"],
            symbol=data["symbol"],
            config=data.get("config", {}),
            metrics=metrics,
            trades=trades,
        )


class RunArchive:
    """Manages run archives for self-improvement analysis."""

    def __init__(self, runs_dir: Path):
        """Initialize archive.

        Args:
            runs_dir: Directory for storing run JSON files
        """
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._current_run: RunRecord | None = None

    def start_run(self, symbol: str, config: dict, mode: str = "paper") -> str:
        """Start a new run.

        Args:
            symbol: Trading symbol
            config: Strategy configuration
            mode: Trading mode ("paper" or "live")

        Returns:
            Run ID
        """
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"

        self._current_run = RunRecord(
            run_id=run_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            mode=mode,
            symbol=symbol,
            config=config,
            metrics=None,
            trades=[],
        )

        return run_id

    def record_trade(
        self,
        action: str,
        symbol: str,
        price: float,
        qty: float,
        reason: str,
        pnl: float | None = None,
        poc: float | None = None,
        va_high: float | None = None,
        va_low: float | None = None,
    ) -> None:
        """Record a trade.

        Args:
            action: Trade action ("buy" or "sell")
            symbol: Stock symbol
            price: Execution price
            qty: Quantity
            reason: Signal reason
            pnl: P&L (for closing trades)
            poc: POC at time of trade
            va_high: Value Area High
            va_low: Value Area Low
        """
        if not self._current_run:
            return

        trade = TradeRecord(
            timestamp=datetime.now().isoformat(),
            action=action,
            symbol=symbol,
            price=price,
            qty=qty,
            reason=reason,
            pnl=pnl,
            poc=poc,
            va_high=va_high,
            va_low=va_low,
        )

        self._current_run.trades.append(trade)

        # Save after each trade
        self._save_current_run()

    def end_run(self) -> RunMetrics | None:
        """End current run and calculate metrics.

        Returns:
            Run metrics
        """
        if not self._current_run:
            return None

        self._current_run.end_time = datetime.now().isoformat()
        self._current_run.metrics = self._calculate_metrics()
        self._save_current_run()

        metrics = self._current_run.metrics
        self._current_run = None

        return metrics

    def _calculate_metrics(self) -> RunMetrics:
        """Calculate metrics for current run."""
        if not self._current_run:
            return RunMetrics(0, 0, 0, 0.0, 0.0, 0.0)

        trades = self._current_run.trades
        closing_trades = [t for t in trades if t.pnl is not None]

        if not closing_trades:
            return RunMetrics(
                total_trades=len(trades),
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                max_drawdown=0.0,
            )

        winning = [t for t in closing_trades if t.pnl and t.pnl > 0]
        losing = [t for t in closing_trades if t.pnl and t.pnl < 0]

        total_pnl = sum(t.pnl or 0 for t in closing_trades)
        win_rate = len(winning) / len(closing_trades) if closing_trades else 0.0

        # Calculate max drawdown
        cumulative_pnl = 0.0
        peak = 0.0
        max_dd = 0.0

        for trade in closing_trades:
            cumulative_pnl += trade.pnl or 0
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            dd = peak - cumulative_pnl
            if dd > max_dd:
                max_dd = dd

        return RunMetrics(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_dd,
        )

    def _save_current_run(self) -> None:
        """Save current run to disk."""
        if not self._current_run:
            return

        file_path = self.runs_dir / f"{self._current_run.run_id}.json"
        with open(file_path, "w") as f:
            json.dump(self._current_run.to_dict(), f, indent=2)

    def load_run(self, run_id: str) -> RunRecord | None:
        """Load a run from disk.

        Args:
            run_id: Run ID

        Returns:
            RunRecord or None if not found
        """
        file_path = self.runs_dir / f"{run_id}.json"
        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)

        return RunRecord.from_dict(data)

    def list_runs(self) -> list[str]:
        """List all run IDs.

        Returns:
            List of run IDs, newest first
        """
        files = sorted(self.runs_dir.glob("*.json"), reverse=True)
        return [f.stem for f in files]

    def get_best_runs(self, metric: str = "sharpe_ratio", top_n: int = 5) -> list[RunRecord]:
        """Get top performing runs by metric.

        Args:
            metric: Metric to sort by
            top_n: Number of runs to return

        Returns:
            List of top runs
        """
        runs = []
        for run_id in self.list_runs():
            run = self.load_run(run_id)
            if run and run.metrics:
                runs.append(run)

        # Sort by metric
        def get_metric(r: RunRecord) -> float:
            if not r.metrics:
                return float("-inf")
            return getattr(r.metrics, metric, 0) or 0

        runs.sort(key=get_metric, reverse=True)
        return runs[:top_n]

    def analyze_parameters(self) -> dict[str, Any]:
        """Analyze which parameters correlate with success.

        Returns:
            Analysis summary for LLM consumption
        """
        runs = [self.load_run(rid) for rid in self.list_runs()]
        runs = [r for r in runs if r and r.metrics]

        if not runs:
            return {"error": "No completed runs to analyze"}

        # Group by parameters and calculate average performance
        param_performance: dict[str, list[float]] = {}

        for run in runs:
            if not run.metrics:
                continue

            for key, value in run.config.items():
                param_key = f"{key}={value}"
                if param_key not in param_performance:
                    param_performance[param_key] = []
                param_performance[param_key].append(run.metrics.win_rate)

        # Calculate averages
        analysis = {
            "total_runs": len(runs),
            "parameter_analysis": {},
        }

        for param, win_rates in param_performance.items():
            analysis["parameter_analysis"][param] = {
                "count": len(win_rates),
                "avg_win_rate": sum(win_rates) / len(win_rates),
            }

        # Find best performing configurations
        best_runs = self.get_best_runs("win_rate", 3)
        analysis["top_configs"] = [
            {
                "run_id": r.run_id,
                "config": r.config,
                "win_rate": r.metrics.win_rate if r.metrics else 0,
            }
            for r in best_runs
        ]

        return analysis

"""
Strategy Optimization Framework using JMetalPy, WFO, and Portfolio Engine
Base classes for strategy optimization with support for different:
- Trading strategies
- Data loaders
- Optimization algorithms
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
import logging

from portfolio_engine import (
    PortfolioEngine,
    OrderType,
    OrderSide,
    Bar,
    AlpacaStockPortfolioEngine,
    AlpacaCryptoPortfolioEngine,
    CoinbasePortfolioEngine,
)
from base_data_loader import BaseDataLoader
from config_manager import Config

# Disable JMetalPy debug logging
logging.getLogger('jmetal').setLevel(logging.WARNING)
logging.getLogger('jmetal.core.algorithm').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class StrategyParameters:
    """Base class for strategy parameters"""

    pass


@dataclass
class OptimizationWindow:
    """Represents a single optimization window for WFO"""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days


class DataLoaderAdapter:
    """Adapter to use BaseDataLoader implementations with the optimization framework"""

    def __init__(self, base_loader: BaseDataLoader, use_transformed: bool = False, hybrid_mode: bool = False):
        """
        Initialize with a BaseDataLoader instance

        Args:
            base_loader: An instance of BaseDataLoader (e.g., AlpacaDataLoader)
            use_transformed: If True, load .transform.csv files with fd/lr columns
            hybrid_mode: If True, load both raw and transformed data (raw for prices, transformed as features)
        """
        self.base_loader = base_loader
        self.config = base_loader.config
        self.use_transformed = use_transformed
        self.hybrid_mode = hybrid_mode

    def load_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Load market data for given symbol and date range

        Returns:
        --------
        pd.DataFrame with columns: open, high, low, close, volume, timestamp
        """
        # Update config symbol if different
        original_symbol = self.config.symbol
        if symbol != original_symbol:
            self.config.symbol = symbol

        try:
            # Use fetch_bars to get data for specific date range
            df = self.base_loader.fetch_bars(start=start_date, end=end_date)

            # Debug: Log data info
            logger.info(
                f"Loaded {len(df)} bars for {symbol} from {start_date} to {end_date}"
            )
            if len(df) > 0:
                logger.debug(
                    f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}"
                )
                logger.debug(
                    f"First close: {df['close'].iloc[0]}, Last close: {df['close'].iloc[-1]}"
                )

            # Ensure we have the required columns
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(
                    f"Missing required columns. Got: {df.columns.tolist()}"
                )

            # Handle different data loading modes
            if self.hybrid_mode:
                # Load transformed features and merge with raw data
                transformed_features = self._load_transformed_features(symbol, df)
                if transformed_features is not None:
                    df = df.merge(transformed_features, on='timestamp', how='left')
                    logger.info(f"Loaded hybrid data with raw prices and transformed features")
                else:
                    logger.warning(f"No transformed features found, using raw data only")
            elif self.use_transformed:
                # Replace raw data with transformed data
                transformed_df = self._load_transformed_data(symbol, df)
                if transformed_df is not None:
                    return transformed_df
                else:
                    logger.warning(f"Falling back to raw data for {symbol}")

            return df
        finally:
            # Restore original symbol
            self.config.symbol = original_symbol

    def _load_transformed_data(self, symbol: str, original_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Load transformed data from .transform.csv file"""
        try:
            import os
            
            # Convert symbol format (e.g., "BTC/USD" -> "btc_usd")
            symbol_formatted = symbol.replace("/", "_").lower()
            
            # Look for transformed file in data directory
            data_dir = "data"
            transform_file = os.path.join(data_dir, f"{symbol_formatted}_{self.config.timeframe}.transform.csv")
            
            if not os.path.exists(transform_file):
                logger.debug(f"Transform file not found: {transform_file}")
                return None
                
            # Load transformed data
            logger.info(f"Loading transformed data from: {transform_file}")
            transformed_df = pd.read_csv(transform_file)
            
            # Convert timestamp if it's a string
            if 'timestamp' in transformed_df.columns:
                transformed_df['timestamp'] = pd.to_datetime(transformed_df['timestamp'])
            else:
                # If no timestamp, assume index matches original
                transformed_df['timestamp'] = original_df['timestamp'].values
                
            # Verify required columns exist
            required_cols = ['open_fd', 'high_fd', 'low_fd', 'close_fd', 'volume_lr']
            if not all(col in transformed_df.columns for col in required_cols):
                logger.warning(f"Missing required transformed columns. Found: {transformed_df.columns.tolist()}")
                return None
                
            # Map transformed columns to standard names
            result_df = pd.DataFrame({
                'timestamp': transformed_df['timestamp'],
                'open': transformed_df['open_fd'],
                'high': transformed_df['high_fd'],
                'low': transformed_df['low_fd'],
                'close': transformed_df['close_fd'],
                'volume': transformed_df['volume_lr']
            })
            
            # No filtering - use all available transformed data
            # The date range will be handled by the calling function
            
            logger.info(f"Using transformed data: {len(result_df)} bars with fd/lr columns")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error loading transformed data: {e}")
            return None
    
    def _load_transformed_features(self, symbol: str, original_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Load transformed features for hybrid mode"""
        try:
            import os
            
            # Convert symbol format
            symbol_formatted = symbol.replace("/", "_").lower()
            
            # Look for transformed file
            data_dir = "data"
            transform_file = os.path.join(data_dir, f"{symbol_formatted}_{self.config.timeframe}.transform.csv")
            
            if not os.path.exists(transform_file):
                logger.debug(f"Transform file not found: {transform_file}")
                return None
                
            # Load transformed data
            logger.info(f"Loading transformed features from: {transform_file}")
            transformed_df = pd.read_csv(transform_file)
            
            # Convert timestamp
            if 'timestamp' in transformed_df.columns:
                transformed_df['timestamp'] = pd.to_datetime(transformed_df['timestamp'])
            
            # Select only the transformed features
            feature_cols = ['timestamp', 'open_fd', 'high_fd', 'low_fd', 'close_fd', 'volume_lr']
            if all(col in transformed_df.columns for col in feature_cols):
                return transformed_df[feature_cols]
            else:
                logger.warning(f"Missing required feature columns in transform file")
                return None
                
        except Exception as e:
            logger.error(f"Error loading transformed features: {e}")
            return None

    def create_bars(self, df: pd.DataFrame) -> List[Bar]:
        """Convert DataFrame to list of Bar objects"""
        bars = []
        
        # Check if we have transformed features
        has_features = any(col in df.columns for col in ['open_fd', 'high_fd', 'low_fd', 'close_fd', 'volume_lr'])
        
        for _, row in df.iterrows():
            bar = Bar(
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                timestamp=row["timestamp"],
            )
            
            # Add features if in hybrid mode
            if has_features:
                bar.features = {
                    'open_fd': row.get('open_fd', None),
                    'high_fd': row.get('high_fd', None),
                    'low_fd': row.get('low_fd', None),
                    'close_fd': row.get('close_fd', None),
                    'volume_lr': row.get('volume_lr', None)
                }
                
            bars.append(bar)
        return bars


class Strategy(ABC):
    """Abstract base class for trading strategies"""

    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter names and their bounds for optimization"""
        pass

    @abstractmethod
    def create_parameters(self, values: List[float]) -> StrategyParameters:
        """Create parameter object from optimization values"""
        pass

    @abstractmethod
    def generate_signals(
        self, bars: List[Bar], params: StrategyParameters
    ) -> List[Tuple[datetime, OrderSide, float]]:
        """
        Generate trading signals from bars and parameters

        Returns:
        --------
        List of (timestamp, side, quantity) tuples
        """
        pass


class StrategyEvaluator:
    """Evaluates strategy performance using portfolio engine"""

    def __init__(
        self,
        strategy: Strategy,
        portfolio_engine_class: type = PortfolioEngine,
        initial_balance: float = 100000.0,
    ):
        self.strategy = strategy
        self.portfolio_engine_class = portfolio_engine_class
        self.initial_balance = initial_balance

    def evaluate(self, bars: List[Bar], params: StrategyParameters) -> Dict[str, float]:
        """
        Evaluate strategy performance

        Returns:
        --------
        Dictionary of performance metrics
        """
        # Create fresh portfolio engine
        engine = self.portfolio_engine_class(initial_balance=self.initial_balance)

        # Debug: Check if we have bars
        if not bars:
            logger.warning("No bars provided for evaluation")
            return engine.get_performance_metrics()

        # Generate signals
        signals = self.strategy.generate_signals(bars, params)

        # Debug: Log signal count
        logger.debug(f"Generated {len(signals)} signals from {len(bars)} bars")

        # Process bars and execute signals
        signal_idx = 0
        for bar in bars:
            # Check if we have a signal for this timestamp
            while signal_idx < len(signals) and signals[signal_idx][0] <= bar.timestamp:
                timestamp, side, position_pct = signals[signal_idx]

                # Calculate actual quantity based on side
                if side == OrderSide.BUY:
                    # Calculate how much we can buy with available cash
                    available_cash = engine.cash * position_pct
                    # Account for commission
                    commission_rate = engine.commission_rate
                    max_quantity = available_cash / (bar.close * (1 + commission_rate))
                    quantity = max_quantity * 0.99  # Use 99% to avoid rounding issues
                else:  # SELL
                    # Sell entire position
                    if engine.position:
                        quantity = engine.position.quantity
                    else:
                        signal_idx += 1
                        continue

                # Place order
                if quantity > 0:
                    engine.place_order(
                        side=side, quantity=quantity, order_type=OrderType.MARKET
                    )
                signal_idx += 1

            # Process bar
            engine.process_bar(bar)

        # Liquidate at end
        engine.liquidate()
        if bars:
            engine.process_bar(bars[-1])

        # Get performance metrics
        return engine.get_performance_metrics()


class StrategyOptimizationProblem(Problem):
    """JMetal problem for strategy optimization"""

    def __init__(
        self,
        strategy: Strategy,
        train_bars: List[Bar],
        evaluator: StrategyEvaluator,
        objectives: List[str] | None = None,
    ):
        # Get parameter bounds first
        self.strategy = strategy
        self.train_bars = train_bars
        self.evaluator = evaluator

        # Default objectives: maximize return, minimize drawdown
        self.objectives = objectives or ["total_return_pct", "max_drawdown_pct"]

        # Get parameter bounds
        self.param_bounds = strategy.get_parameter_bounds()
        self.param_names = list(self.param_bounds.keys())

        # Set bounds
        self.lower_bound = [self.param_bounds[name][0] for name in self.param_names]
        self.upper_bound = [self.param_bounds[name][1] for name in self.param_names]

        # Objective directions (minimize by default in JMetal)
        # We'll handle maximization by negating values
        self.obj_directions = []
        for obj in self.objectives:
            if obj in [
                "total_return_pct",
                "win_rate",
                "profit_factor",
                "sharpe_ratio",
                "sortino_ratio",
            ]:
                self.obj_directions.append(-1)  # Maximize (negate for minimization)
            else:
                self.obj_directions.append(1)  # Minimize

        # Initialize parent after setting up our attributes
        super().__init__()

    @property
    def number_of_variables(self) -> int:
        """Return number of variables"""
        return len(self.param_names)

    @property
    def number_of_objectives(self) -> int:
        """Return number of objectives"""
        return len(self.objectives)

    @property
    def number_of_constraints(self) -> int:
        """Return number of constraints"""
        return 0

    @property
    def name(self) -> str:
        """Return problem name"""
        return f"Strategy Optimization ({self.strategy.__class__.__name__})"

    def create_solution(self) -> FloatSolution:
        """Create a new solution"""
        solution = FloatSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives,
            number_of_constraints=self.number_of_constraints,
        )
        # Initialize variables with random values within bounds
        import random

        solution.variables = []
        for i in range(self.number_of_variables):
            solution.variables.append(
                random.uniform(self.lower_bound[i], self.upper_bound[i])
            )
        return solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        """Evaluate a solution"""
        # Create parameters from solution
        # Debug: print what we're getting
        # print(f"DEBUG: solution.variables = {solution.variables}, type = {type(solution.variables)}")
        params = self.strategy.create_parameters(solution.variables)

        # Evaluate strategy
        metrics = self.evaluator.evaluate(self.train_bars, params)

        # Set objectives
        solution.objectives = []
        for i, obj_name in enumerate(self.objectives):
            value = metrics.get(obj_name, 0.0)
            # Apply direction (negate for maximization)
            solution.objectives.append(value * self.obj_directions[i])

        return solution


class WalkForwardOptimizer:
    """Walk-Forward Optimization implementation"""

    def __init__(
        self,
        strategy: Strategy,
        data_loader: DataLoaderAdapter,
        portfolio_engine_class: type = PortfolioEngine,
        initial_balance: float = 100000.0,
    ):
        self.strategy = strategy
        self.data_loader = data_loader
        self.portfolio_engine_class = portfolio_engine_class
        self.initial_balance = initial_balance
        self.evaluator = StrategyEvaluator(
            strategy, portfolio_engine_class, initial_balance
        )

    def create_windows(
        self,
        start_date: datetime,
        end_date: datetime,
        train_days: int = 252,  # 1 year
        test_days: int = 63,  # 3 months
        step_days: int = 21,
    ) -> List[OptimizationWindow]:  # 1 month
        """Create optimization windows for WFO"""
        windows = []
        current_start = start_date

        while current_start < end_date:
            train_end = current_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            # Check if we have enough data
            if test_end > end_date:
                break

            window = OptimizationWindow(
                train_start=current_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            windows.append(window)

            # Move to next window
            current_start += timedelta(days=step_days)

        return windows

    def optimize_window(
        self,
        window: OptimizationWindow,
        symbol: str,
        algorithm_factory=None,
        max_evaluations: int = 1000,
    ) -> Tuple[StrategyParameters, Dict[str, float]]:
        """
        Optimize strategy for a single window

        Returns:
        --------
        Tuple of (best_parameters, out_of_sample_metrics)
        """
        # Load training data
        train_df = self.data_loader.load_data(
            symbol, window.train_start, window.train_end
        )
        train_bars = self.data_loader.create_bars(train_df)

        # Create optimization problem
        problem = StrategyOptimizationProblem(
            strategy=self.strategy, train_bars=train_bars, evaluator=self.evaluator
        )

        # Create algorithm
        if algorithm_factory is None:
            algorithm = NSGAII(
                problem=problem,
                population_size=100,
                offspring_population_size=100,
                mutation=PolynomialMutation(
                    probability=1.0 / problem.number_of_variables, distribution_index=20
                ),
                crossover=SBXCrossover(probability=0.9, distribution_index=20),
                termination_criterion=StoppingByEvaluations(
                    max_evaluations=max_evaluations
                ),
            )
        else:
            algorithm = algorithm_factory(problem)

        # Run optimization
        algorithm.run()

        # Get non-dominated solutions
        solutions = get_non_dominated_solutions(algorithm.result())

        # Select best solution (highest return)
        best_solution = None
        best_return = float("-inf")

        for solution in solutions:
            # Get actual return value (remember we negated for minimization)
            return_idx = problem.objectives.index("total_return_pct")
            actual_return = -solution.objectives[return_idx]

            if actual_return > best_return:
                best_return = actual_return
                best_solution = solution

        # Create best parameters
        best_params = self.strategy.create_parameters(best_solution.variables)

        # Evaluate on test data
        test_df = self.data_loader.load_data(symbol, window.test_start, window.test_end)
        test_bars = self.data_loader.create_bars(test_df)

        out_of_sample_metrics = self.evaluator.evaluate(test_bars, best_params)

        return best_params, out_of_sample_metrics

    def run(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 21,
        max_evaluations: int = 1000,
    ) -> List[Tuple[OptimizationWindow, StrategyParameters, Dict[str, float]]]:
        """
        Run full walk-forward optimization

        Returns:
        --------
        List of (window, best_parameters, out_of_sample_metrics) tuples
        """
        # Create windows
        windows = self.create_windows(
            start_date=start_date,
            end_date=end_date,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
        )

        logger.info(f"Created {len(windows)} optimization windows")

        # Optimize each window
        results = []
        for i, window in enumerate(windows):
            logger.info(
                f"Optimizing window {i+1}/{len(windows)}: "
                f"{window.train_start.date()} to {window.test_end.date()}"
            )

            best_params, oos_metrics = self.optimize_window(
                window=window, symbol=symbol, max_evaluations=max_evaluations
            )

            results.append((window, best_params, oos_metrics))

            logger.info(
                f"Window {i+1} OOS Return: {oos_metrics['total_return_pct']:.2f}%, "
                f"Sharpe: {oos_metrics['sharpe_ratio']:.2f}"
            )

        return results

    def analyze_results(
        self,
        results: List[Tuple[OptimizationWindow, StrategyParameters, Dict[str, float]]],
    ) -> Dict[str, float]:
        """Analyze WFO results"""
        # Aggregate out-of-sample metrics
        oos_returns = []
        oos_sharpes = []
        oos_drawdowns = []
        win_rates = []

        for window, params, metrics in results:
            oos_returns.append(metrics["total_return_pct"])
            oos_sharpes.append(metrics["sharpe_ratio"])
            oos_drawdowns.append(metrics["max_drawdown_pct"])
            win_rates.append(metrics["win_rate"])

        # Calculate aggregate statistics
        analysis = {
            "num_windows": len(results),
            "avg_oos_return": np.mean(oos_returns),
            "std_oos_return": np.std(oos_returns),
            "min_oos_return": np.min(oos_returns),
            "max_oos_return": np.max(oos_returns),
            "avg_sharpe": np.mean(oos_sharpes),
            "avg_max_drawdown": np.mean(oos_drawdowns),
            "avg_win_rate": np.mean(win_rates),
            "positive_windows": sum(1 for r in oos_returns if r > 0),
            "negative_windows": sum(1 for r in oos_returns if r < 0),
            "win_ratio": (
                sum(1 for r in oos_returns if r > 0) / len(oos_returns)
                if oos_returns
                else 0
            ),
        }

        return analysis

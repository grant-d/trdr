#!/usr/bin/env python3
"""
Main trading loop for automated strategy optimization.

This is the core loop that:
1. Loads/catches up market data
2. Generates dollar bars
3. Runs WFO + GA optimization
4. Logs best parameters
5. Pauses and loops
"""

import time
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from alpaca_data_loader import AlpacaDataLoader
from data_pipeline import DataPipeline
from config_manager import Config
from optimizer import Optimizer, OptimizerConfig
from strategy_parameters import BaseStrategyParameters, IntegerRange, MinMaxRange
from bar_aggregators import DollarBarAggregator
from pydantic import BaseModel, Field


class SplitResult(BaseModel):
    """Result of a single optimization split."""
    split_id: int
    train_fitness: float
    test_performance: float
    params: Dict[str, Any]


class TradingLoopResult(BaseModel):
    """Result of optimization run."""
    consensus_params: Dict[str, Any]
    weighted_params: Dict[str, Any]
    avg_test_performance: float
    num_splits: int
    individual_results: List[SplitResult]


class TradingStrategyParameters(BaseStrategyParameters):
    """Example trading strategy parameters."""
    
    fast_ma: IntegerRange = Field(
        default_factory=lambda: IntegerRange(min_value=5, max_value=50),
        description="Fast moving average period"
    )
    slow_ma: IntegerRange = Field(
        default_factory=lambda: IntegerRange(min_value=20, max_value=200),
        description="Slow moving average period"
    )
    entry_threshold: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=0.001, max_value=0.01),
        description="Entry signal threshold"
    )
    
    def validate_constraints(self) -> bool:
        """Ensure fast MA is shorter than slow MA."""
        # Sample values to check constraints
        fast_val = self.fast_ma.sample()
        slow_val = self.slow_ma.sample()
        return fast_val < slow_val


def simple_ma_crossover_fitness(params: TradingStrategyParameters, data: pd.DataFrame) -> float:
    """
    Simple moving average crossover strategy fitness function.
    
    Args:
        params: TradingStrategyParameters with strategy parameters
        data: Market data with OHLCV columns
        
    Returns:
        Sharpe ratio as fitness score
    """
    param_dict = params.to_dict()
    fast_ma = int(param_dict['fast_ma'])
    slow_ma = int(param_dict['slow_ma'])
    threshold = param_dict['entry_threshold']
    
    # Ensure fast < slow
    if fast_ma >= slow_ma:
        return -1000.0
        
    # Calculate moving averages
    data = data.copy()
    data['fast_ma'] = data['close'].rolling(window=fast_ma).mean()
    data['slow_ma'] = data['close'].rolling(window=slow_ma).mean()
    
    # Generate signals with threshold
    data['ma_diff'] = (data['fast_ma'] - data['slow_ma']) / data['slow_ma']
    data['signal'] = 0
    data.loc[data['ma_diff'] > threshold, 'signal'] = 1
    data.loc[data['ma_diff'] < -threshold, 'signal'] = -1
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
    
    # Drop NaN values
    data = data.dropna()
    
    if len(data) < 20:
        return -1000.0
        
    # Calculate Sharpe ratio
    mean_return = data['strategy_returns'].mean()
    std_return = data['strategy_returns'].std()
    
    if std_return == 0:
        return 0.0
        
    # Annualize based on data frequency (assume 1m bars)
    periods_per_year = 252 * 390  # 252 trading days * 390 minutes per day
    sharpe_ratio = (mean_return / std_return) * (periods_per_year ** 0.5)
    
    return float(sharpe_ratio)


class TradingLoop:
    """Main trading loop coordinator."""
    
    def __init__(self, config_path: str):
        """Initialize the trading loop."""
        self.config_path = f"configs/{config_path}"
        self.config = Config(self.config_path)
        
        # Initialize components
        self.data_loader = AlpacaDataLoader(self.config)
        self.data_pipeline = DataPipeline()
        
        # Best parameters tracking
        self.best_params_file = Path(".taskmaster/logs/best_params.json")
        self.best_params_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize best parameters log
        self.best_params_log = self._load_best_params_log()
        
    def _load_best_params_log(self) -> Dict[str, Any]:
        """Load existing best parameters log."""
        if self.best_params_file.exists():
            try:
                with open(self.best_params_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load best params log: {e}")
                
        return {
            "optimization_history": [],
            "current_best": None,
            "best_fitness": -float('inf'),
            "last_updated": None
        }
    
    def _save_best_params_log(self):
        """Save best parameters log to file."""
        try:
            with open(self.best_params_file, 'w') as f:
                json.dump(self.best_params_log, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save best params log: {e}")
    
    def _update_best_params(self, params: Dict[str, Any], fitness: float, timestamp: datetime):
        """Update best parameters if new fitness is better."""
        if fitness > self.best_params_log["best_fitness"]:
            self.best_params_log["current_best"] = params
            self.best_params_log["best_fitness"] = fitness
            self.best_params_log["last_updated"] = timestamp
            
            print(f"🎯 NEW BEST PARAMETERS! Fitness: {fitness:.4f}")
            print(f"   Parameters: {params}")
            
        # Always log to history
        self.best_params_log["optimization_history"].append({
            "timestamp": timestamp,
            "parameters": params,
            "fitness": fitness,
            "is_best": fitness == self.best_params_log["best_fitness"]
        })
        
        # Keep only last 100 entries
        if len(self.best_params_log["optimization_history"]) > 100:
            self.best_params_log["optimization_history"] = self.best_params_log["optimization_history"][-100:]
    
    def _load_and_catchup_data(self) -> pd.DataFrame:
        """Load existing data and catch up with latest."""
        print("\n📊 Loading and catching up data...")
        
        # Load existing data
        df = self.data_loader.load_data()
        
        if df.empty:
            print("   No existing data found, starting fresh")
            return df
            
        print(f"   Loaded {len(df)} existing bars")
        print(f"   Latest data: {df['timestamp'].max()}")
        
        # Check if we need to catch up
        now = datetime.now()
        latest_data_time = pd.to_datetime(df['timestamp'].max())
        
        # Make timezone-aware for comparison
        if latest_data_time.tz is not None:
            now = now.replace(tzinfo=latest_data_time.tz)
        else:
            latest_data_time = latest_data_time.tz_localize(None)
            
        if now - latest_data_time > timedelta(minutes=5):
            print("   Catching up with latest data...")
            # In a real system, you'd fetch new data here
            # For now, we'll just use existing data
            
        return df
    
    def _generate_dollar_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate dollar bars from the raw data."""
        print("\n💰 Generating dollar bars...")
        
        if df.empty:
            print("   No data to process")
            return pd.DataFrame()
            
        # Enable pipeline and dollar bars in config
        self.config.pipeline.enabled = True
        self.config.pipeline.dollar_bars.enabled = True
        
        # Use existing threshold or calculate one if not set
        if self.config.pipeline.dollar_bars.threshold is None:
            self.config.pipeline.dollar_bars.threshold = DollarBarAggregator.estimate_threshold(df, target_bars=1000)
        else:
            print(f"   Using existing threshold: ${self.config.pipeline.dollar_bars.threshold:.2f}")
        
        # Process through pipeline directly
        cleaned_df, dollar_bars_df = self.data_pipeline.process(
            df=df,
            zero_volume_keep_percentage=self.config.pipeline.zero_volume_keep_percentage,
            dollar_bar_threshold=self.config.pipeline.dollar_bars.threshold,
            price_column=self.config.pipeline.dollar_bars.price_column
        )
        
        if dollar_bars_df is None or dollar_bars_df.empty:
            print("   No dollar bars generated")
            return pd.DataFrame()
            
        print(f"   Generated {len(dollar_bars_df)} dollar bars")
        return dollar_bars_df
    
    def _run_optimization(self, dollar_bars: pd.DataFrame) -> Optional[TradingLoopResult]:
        """Run WFO + GA optimization on dollar bars."""
        print("\n🧬 Running WFO + GA optimization...")
        
        # For dollar bars, we need fewer bars since they're more information-dense
        min_bars_required = 100  # Much lower than time bars since dollar bars are compressed
        
        if len(dollar_bars) < min_bars_required:
            print(f"   Insufficient data for optimization: {len(dollar_bars)} bars (need {min_bars_required})")
            return None
            
        # Configure optimizer
        optimizer_config = OptimizerConfig(
            n_splits=self.config.optimizer.n_splits,
            test_ratio=self.config.optimizer.test_ratio,  # Pass percentage directly
            gap=0,
            param_class=TradingStrategyParameters,
            population_size=self.config.optimizer.population_size,
            generations=self.config.optimizer.generations,
            crossover_prob=0.8,
            mutation_prob=0.2,
            tournament_size=3,
            verbose=True,
            seed=42
        )
        
        # Create optimizer
        optimizer = Optimizer(optimizer_config)
        
        # Run optimization
        results = optimizer.optimize(
            data=dollar_bars,
            fitness_function=simple_ma_crossover_fitness,
            evaluation_function=simple_ma_crossover_fitness
        )
        
        if not results:
            print("   No optimization results")
            return None
            
        # Get best parameters
        consensus_params = optimizer.get_consensus_parameters()
        weighted_params = optimizer.get_performance_weighted_parameters()
        
        # Calculate overall performance
        test_performances = [r.test_performance for r in results]
        avg_test_performance = sum(test_performances) / len(test_performances)
        
        best_result = TradingLoopResult(
            consensus_params=consensus_params,
            weighted_params=weighted_params,
            avg_test_performance=avg_test_performance,
            num_splits=len(results),
            individual_results=[
                SplitResult(
                    split_id=r.split_id,
                    train_fitness=r.train_fitness,
                    test_performance=r.test_performance,
                    params=r.best_params.to_dict()
                )
                for r in results
            ]
        )
        
        print(f"   Optimization complete. Avg test performance: {avg_test_performance:.4f}")
        return best_result


    def run_iteration(self):
        """Run a single iteration of the trading loop."""
        print(f"\n{'='*80}")
        print(f"🚀 TRADING LOOP ITERATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        try:
            # 1. Load and catchup data
            raw_data = self._load_and_catchup_data()
            
            if raw_data.empty:
                print("⚠️  No data available, skipping iteration")
                return
                
            # 2. Generate dollar bars
            dollar_bars = self._generate_dollar_bars(raw_data)
            
            if dollar_bars.empty:
                print("⚠️  No dollar bars generated, skipping iteration")
                return
                
            # 3. Run WFO + GA optimization
            optimization_result = self._run_optimization(dollar_bars)
            
            if optimization_result is None:
                print("⚠️  Optimization failed, skipping iteration")
                return
                
            # 4. Log best parameters
            timestamp = datetime.now()
            self._update_best_params(
                optimization_result.consensus_params,
                optimization_result.avg_test_performance,
                timestamp
            )
            
            # Save to file
            self._save_best_params_log()
            
            print(f"\n✅ Iteration complete at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"\n❌ Error in iteration: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_pause_time(self, user_pause_minutes: Optional[int] = None) -> int:
        """Calculate appropriate pause time based on timeframe."""
        # Timeframe to minutes mapping
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080
        }
        
        tf_minutes = timeframe_minutes.get(self.config.timeframe, 60)
        
        # If user specified a pause time, use it but warn if it's too long
        if user_pause_minutes is not None:
            if user_pause_minutes >= tf_minutes:
                print(f"⚠️  Warning: Pause time ({user_pause_minutes}m) >= timeframe ({tf_minutes}m)")
                print(f"   This may cause missed bars!")
            return user_pause_minutes
        
        # Calculate optimal pause time (slightly less than timeframe)
        # For very short timeframes, use a fraction
        if tf_minutes <= 5:
            return max(1, int(tf_minutes * 0.8))  # 80% of timeframe, min 1 minute
        else:
            return max(1, tf_minutes - 2)  # 2 minutes before next bar
    
    def run_loop(self, pause_minutes: Optional[int] = None):
        """Run the main trading loop."""
        actual_pause = self._calculate_pause_time(pause_minutes)
        
        print(f"\n🔄 Starting trading loop")
        print(f"   Config: {self.config_path}")
        print(f"   Symbol: {self.config.symbol}")
        print(f"   Timeframe: {self.config.timeframe}")
        print(f"   Pause time: {actual_pause} minutes (optimized for {self.config.timeframe} bars)")
        
        if pause_minutes and pause_minutes != actual_pause:
            print(f"   User requested: {pause_minutes} minutes")
        
        # Show current best if available
        if self.best_params_log["current_best"]:
            print(f"   Current best fitness: {self.best_params_log['best_fitness']:.4f}")
            print(f"   Current best params: {self.best_params_log['current_best']}")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n🔄 ITERATION {iteration}")
                
                # Run one iteration
                self.run_iteration()
                
                # Pause before next iteration
                print(f"\n⏸️  Pausing for {actual_pause} minutes...")
                print(f"   Next iteration at: {(datetime.now() + timedelta(minutes=actual_pause)).strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(actual_pause * 60)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Trading loop stopped by user")
            
        except Exception as e:
            print(f"\n❌ Trading loop error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            print(f"\n📋 Final best parameters:")
            if self.best_params_log["current_best"]:
                print(f"   Fitness: {self.best_params_log['best_fitness']:.4f}")
                print(f"   Parameters: {self.best_params_log['current_best']}")
            else:
                print("   No best parameters found")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading loop with WFO + GA optimization")
    parser.add_argument("--config", default="btc_usd_1m_config.json", help="Configuration file")
    parser.add_argument("--optimize", action="store_true", help="Run optimization loop")
    parser.add_argument("--pause", type=int, help="Pause between iterations (minutes, auto-calculated if not specified)")
    parser.add_argument("--single", action="store_true", help="Run single iteration only")
    
    args = parser.parse_args()
    
    # Only run optimization if --optimize flag is provided
    if not args.optimize:
        print("Trading loop optimization disabled. Use --optimize flag to enable.")
        return
    
    # Create trading loop
    loop = TradingLoop(args.config)
    
    if args.single:
        # Run single iteration
        loop.run_iteration()
    else:
        # Run continuous loop
        loop.run_loop(pause_minutes=args.pause)


if __name__ == "__main__":
    main()
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
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import random
from alpaca_data_loader import AlpacaDataLoader
from data_pipeline import DataPipeline
from config_manager import Config
from optimizer import Optimizer, OptimizerConfig
from strategy_parameters import BaseStrategyParameters, IntegerRange, MinMaxRange
from regime_strategy import RegimeStrategyParameters
from regime_strategy import RegimeStrategy
from indicators import calculate_mss
from backtest_simulator import BacktestSimulator
from bar_aggregators import DollarBarAggregator
from pydantic import BaseModel, Field
from colors import header, title, success, warning, error, info, value, highlight, section, separator, format_number, format_percentage
from performance import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_calmar_ratio
from state import RuntimeState, ParameterValue

class SplitResult(BaseModel):
    """Result of a single optimization split."""
    split_id: int
    train_fitness: float
    test_performance: float
    params: Dict[str, ParameterValue]


class TradingLoopResult(BaseModel):
    """Result of optimization run."""
    consensus_params: Dict[str, ParameterValue]
    weighted_params: Dict[str, ParameterValue]
    avg_test_performance: float
    num_splits: int
    individual_results: List[SplitResult]


# Use regime strategy parameters instead of MA crossover
TradingStrategyParameters = RegimeStrategyParameters


def regime_strategy_fitness(params: RegimeStrategyParameters, data: pd.DataFrame) -> float:
    """
    Regime-based trading strategy fitness function.
    
    Args:
        params: RegimeStrategyParameters with strategy parameters
        data: Market data with OHLCV columns
        
    Returns:
        Blended fitness score combining Sharpe, Sortino, Calmar ratios and transaction cost penalty
    """
    # Validate constraints
    if not params.validate_constraints():
        return -1000.0

    # Get parameter dictionary
    param_dict = params.to_dict()
    
    # Calculate market state indicators
    factors_df, regimes = calculate_mss(data)
    
    # Ensure we have enough data
    factors_df = factors_df.dropna()
    if len(factors_df) < 50:
        return -1000.0
    
    # Initialize strategy with parameters
    strategy = RegimeStrategy(
        initial_capital=100000.0,
        max_position_fraction=param_dict['max_position_fraction'],
        entry_step_size=param_dict['entry_step_size'],
        stop_loss_multiplier_strong=param_dict['stop_loss_multiplier_strong'],
        stop_loss_multiplier_weak=param_dict['stop_loss_multiplier_weak'],
        strong_bull_threshold=param_dict['strong_bull_threshold'],
        weak_bull_threshold=param_dict['weak_bull_threshold'],
        neutral_upper=param_dict['neutral_upper'],
        neutral_lower=param_dict['neutral_lower'],
        weak_bear_threshold=param_dict['weak_bear_threshold'],
        strong_bear_threshold=param_dict['strong_bear_threshold'],
        allow_shorts=False  # Conservative approach for now
    )
    
    # Run backtest
    results_df = strategy.run_backtest(data, factors_df)
    if len(results_df) < 20:
        return -1000.0
    
    # Calculate returns from portfolio value
    results_df['returns'] = results_df['portfolio_value'].pct_change()
    results_df = results_df.dropna()
    
    if len(results_df) < 20:
        return -1000.0
    
    # Calculate transaction costs
    trade_costs = 0.0025  # Alpaca's 0.25% commission
    position_changes = results_df['position_units'].diff().abs()
    trade_values = position_changes * results_df['close']
    total_costs = (trade_values * trade_costs).sum()
    cost_percentage = total_costs / 100000.0  # As percentage of initial capital
    
    # Hard cutoff for extreme overtrading
    if cost_percentage > 0.05:  # More than 5% of capital in costs
        return -1000.0
    
    # Apply transaction costs to returns
    cost_per_period = total_costs / results_df['portfolio_value'].mean() / len(results_df)
    cost_adjusted_returns = results_df['returns'] - cost_per_period
    
    # Handle zero volatility case
    if cost_adjusted_returns.std() < 1e-10:
        if len(results_df) > 0 and results_df['portfolio_value'].iloc[0] > 0:
            total_return = (results_df['portfolio_value'].iloc[-1] / results_df['portfolio_value'].iloc[0]) - 1
            return float(total_return * 100)
        return 0.0
    
    # Estimate annualization factor
    if 'timestamp' in data.columns:
        time_diffs = pd.to_datetime(data['timestamp']).diff().dropna()
        avg_bar_duration = pd.to_timedelta(time_diffs.mean())
        bars_per_year = pd.Timedelta(days=252).total_seconds() / avg_bar_duration.total_seconds()
    else:
        bars_per_year = 252 * 390  # Fallback for minute bars
    
    # Calculate risk-adjusted metrics
    sharpe_ratio = calculate_sharpe_ratio(cost_adjusted_returns, bars_per_year)
    sortino_ratio = calculate_sortino_ratio(cost_adjusted_returns, periods_per_year=int(bars_per_year))
    calmar_ratio = calculate_calmar_ratio(cost_adjusted_returns, periods_per_year=int(bars_per_year))
    
    # Transaction cost penalty (0 to 1, where 1 = no penalty)
    transaction_penalty = max(0.0, 1.0 - (cost_percentage * 20))  # Linear penalty starting at 0%
    
    # Blend metrics with proper weights
    fitness_score = (
        0.4 * sharpe_ratio +
        0.3 * sortino_ratio + 
        0.2 * calmar_ratio +
        0.1 * (transaction_penalty * 10)  # Scale transaction penalty to similar range
    )
    
    # Apply transaction cost penalty as multiplicative factor
    fitness_score *= transaction_penalty
    
    # MSS alignment bonus: reward strategies that trade in alignment with market regimes
    mss_values = results_df['mss'].to_numpy(dtype=float)
    position_values = results_df['position_units'].to_numpy(dtype=float)
    alignment = np.sign(mss_values) * np.sign(position_values)
    aligned_periods = (alignment > 0).sum()
    total_periods = len(results_df)
    alignment_ratio = aligned_periods / total_periods if total_periods > 0 else 0
    
    # Regime quality assessment
    regime_quality = 0
    for idx, row in results_df.iterrows():
        regime = row['regime']
        pos = abs(row['position_units'])
        
        if regime in ['Strong Bull', 'Strong Bear']:
            if pos > 0:
                regime_quality += 1
        elif regime == 'Neutral':
            if pos == 0:
                regime_quality += 1
    
    regime_quality_ratio = regime_quality / len(results_df) if len(results_df) > 0 else 0
    
    # Apply bonuses to fitness score
    if alignment_ratio > 0.6:
        fitness_score *= (1 + 0.1 * (alignment_ratio - 0.6))  # Up to 4% bonus
    
    if regime_quality_ratio > 0.5:
        fitness_score *= (1 + 0.05 * (regime_quality_ratio - 0.5))  # Up to 2.5% bonus
    
    return float(fitness_score)


class TradingLoop:
    """Main trading loop coordinator."""
    
    def __init__(self, config_path: str):
        """Initialize the trading loop."""
        self.config_path = f"configs/{config_path}"
        self.config = Config(self.config_path)
        
        # Initialize components
        self.data_loader = AlpacaDataLoader(self.config)
        self.data_pipeline = DataPipeline()
        
        # Load or create state
        self.state = RuntimeState.load_from_file(self.config.symbol, self.config.timeframe)
        
    def _save_state(self):
        """Save state to file."""
        self.state.save_to_file(self.config.symbol, self.config.timeframe)
    
    def _load_and_catchup_data(self) -> pd.DataFrame:
        """Load existing data and catch up with latest."""
        print(info("\n📊 Loading and catching up data..."))
        
        # Load existing data
        df = self.data_loader.load_data()
        
        if df.empty:
            print(warning("   No existing data found, starting fresh"))
            return df
            
        print(f"   {section('Loaded:')} {highlight(str(len(df)))} existing bars")
        print(f"   {section('Latest data:')} {value(str(df['timestamp'].max()))}")
        
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
        print(info("\n💰 Generating dollar bars..."))
        
        if df.empty:
            print(warning("   No data to process"))
            return pd.DataFrame()
            
        # Enable pipeline and dollar bars in config
        self.config.pipeline.enabled = True
        self.config.pipeline.dollar_bars.enabled = True
        
        # Use existing threshold or calculate one if not set
        if self.config.pipeline.dollar_bars.threshold is None:
            self.config.pipeline.dollar_bars.threshold = DollarBarAggregator.estimate_threshold(df, target_bars=1000)
        else:
            print(f"   {section('Using threshold:')} ${highlight(f"{self.config.pipeline.dollar_bars.threshold:.2f}")}")
        
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
            # test_ratio=self.config.optimizer.test_ratio,  # Pass percentage directly
            gap=0,
            param_class=RegimeStrategyParameters,
            population_size=self.config.optimizer.population_size,
            generations=self.config.optimizer.generations,
            crossover_prob=0.8,
            mutation_prob=0.2,
            tournament_size=3,
            verbose=True,
            seed=random.randint(0, 2**32 - 1)  # Random seed for each run
        )
        
        # Create optimizer
        optimizer = Optimizer(optimizer_config)
        
        # Prepare seed parameters from hall of fame
        seed_params = None
        if self.state.hall_of_fame:
            # Convert hall of fame entries to parameter objects
            seed_params = []
            for entry in self.state.hall_of_fame[:3]:  # Use top 3
                try:
                    params = RegimeStrategyParameters(**entry.parameters)
                    seed_params.append(params)
                except Exception as e:
                    print(f"   Warning: Could not convert hall of fame entry: {e}")
            
            if seed_params:
                print(f"   Seeding with {len(seed_params)} hall of fame members")
        
        # Run optimization
        results = optimizer.optimize(
            data=dollar_bars,
            fitness_function=regime_strategy_fitness,
            evaluation_function=regime_strategy_fitness,
            seed_params=seed_params
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
                
            # 4. Log best parameters - use best individual performance, excluding 0.0
            valid_performances = [r.test_performance for r in optimization_result.individual_results if r.test_performance != 0.0]
            
            if valid_performances:
                best_individual_performance = max(valid_performances)
                is_new_best = self.state.update_best_params(
                    optimization_result.consensus_params,
                    best_individual_performance
                )
            else:
                # All performances were 0.0, skip updating best params
                is_new_best = False
                best_individual_performance = 0.0
            
            if is_new_best:
                print(success("\n🎯 NEW BEST PARAMETERS!"))
                print(f"   {section('Fitness:')} {format_number(best_individual_performance)}")
                print(f"   {section('Parameters:')} {value(str(optimization_result.consensus_params))}")
            
            # Update hall of fame with all good individual results
            for split_result in optimization_result.individual_results:
                if split_result.test_performance > 0:  # Only add profitable strategies
                    self.state.update_hall_of_fame(
                        split_result.params,
                        split_result.test_performance,
                        self.config.symbol
                    )
            
            # Also add consensus parameters if they performed well
            if optimization_result.avg_test_performance > 0:
                self.state.update_hall_of_fame(
                    optimization_result.consensus_params,
                    optimization_result.avg_test_performance,
                    self.config.symbol
                )
            
            # Save state to file
            self._save_state()
            
            # Run backtest simulation with best parameters
            simulator = BacktestSimulator(initial_capital=100000.0)
            simulator.run(
                parameters=optimization_result.consensus_params,
                data=dollar_bars,
                verbose=True
            )
            
            print(f"\n✅ Iteration complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
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
        if self.state.best_params:
            print(f"   Current best fitness: {self.state.best_fitness:.4f}")
            print(f"   Current best params: {self.state.best_params}")
        
        # Show hall of fame status
        if self.state.hall_of_fame:
            print(f"   Hall of Fame: {len(self.state.hall_of_fame)} members (top fitness: {self.state.hall_of_fame[0].fitness:.4f})")
        
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
            if self.state.best_params:
                print(f"   Fitness: {self.state.best_fitness:.4f}")
                print(f"   Parameters: {self.state.best_params}")
            else:
                print("   No best parameters found")
            
            # Save final state
            self._save_state()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading loop with WFO + GA optimization")
    parser.add_argument("--config", default="btc_usd_1m.config.json", help="Configuration file")
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
"""
Live Paper Trading System with CSV Caching
Runs GA optimization on every new bar for dynamic strategy adjustment
Follows exact same workflow as main.py optimize -> test
"""

import time
import argparse
import sys
import os
import platform
import json
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import OrderSide

from alpaca_client import AlpacaClient
from main import handle_optimize_command, handle_test_command
from data_processing import prepare_data, create_dollar_bars, impute_missing_bars
from factors import calculate_mss, calculate_macd, calculate_rsi
from strategy_enhanced import EnhancedTradingStrategy, Position
from chalk import Chalk, green, red, yellow, blue, cyan, magenta, white, bold
from args_types import OptimizeArgs, TestArgs
from portfolio_tracker import PortfolioTracker


class LiveTrader:
    """Live paper trading with continuous optimization and CSV caching"""
    
    def __init__(self, 
                 api_key: str, 
                 secret_key: str, 
                 symbol: str, 
                 is_crypto: bool = False, 
                 timeframe_minutes: int = 1,
                 population: int = 50,
                 generations: int = 20,
                 lookback_bars: int = 200,
                 initial_history_bars: int = 1000,
                 capital: float = 100_000.0,
                 check_interval: int = 5,
                 max_optimization_bars: int = 2000,
                 dollar_threshold: str = "auto",
                 fitness: str = "sortino",
                 allow_shorts: bool = False,
                 quiet_mode: bool = True):
        self.symbol = symbol
        self.is_crypto = is_crypto
        self.timeframe_minutes = timeframe_minutes
        
        # Initialize data clients
        if is_crypto:
            self.data_client = CryptoHistoricalDataClient()
        else:
            self.data_client = StockHistoricalDataClient(api_key, secret_key)
            
        self.alpaca_client = AlpacaClient(
            data_client=self.data_client,
            trade_client=None,  # No trading client needed for simulation
            symbol=symbol
        )
        
        # Parameters matching main.py exactly
        self.population = population
        self.generations = generations
        self.lookback_bars = lookback_bars
        self.initial_history_bars = initial_history_bars
        self.capital = capital
        self.check_interval = check_interval
        self.max_optimization_bars = max_optimization_bars
        self.dollar_threshold = dollar_threshold
        self.fitness = fitness
        self.allow_shorts = allow_shorts
        
        # State tracking
        self.last_bar_time = None
        self.current_strategy = None
        self.current_params = None
        self.historical_data_limit_reached = False  # Track if we've hit the data provider's limit
        self.ga_archive = []  # Archive of top candidates from recent optimizations
        
        # Performance tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.capital]
        self.optimization_history: List[Dict] = []
        
        # Progress tracking for cache messages
        self.cache_fetch_count = 0
        self.cache_message_printed = False
        
        # CSV caching setup
        self.cache_dir = Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Replace slash with underscore for filename
        safe_symbol = symbol.replace("/", "_")
        self.csv_filename = self.cache_dir / f"{safe_symbol}_{timeframe_minutes}min.csv"
        self.cached_df = self.load_cached_data()
        
        # Trading portfolio tracker setup
        self.tracker = PortfolioTracker(symbol, timeframe_minutes, capital)
        
        # Temporary files for main.py integration
        self.temp_dir = Path("./temp_live_trader")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load previous optimization results if available
        self.load_previous_params()
        
    # Portfolio tracking methods delegated to PortfolioTracker class
    @property
    def current_position_size(self) -> float:
        """Get current position size from tracker"""
        return self.tracker.current_position_size
    
    @property
    def available_cash(self) -> float:
        """Get available cash from tracker"""
        return self.tracker.available_cash
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Get total portfolio value from tracker"""
        portfolio = self.tracker.get_portfolio_value(current_price)
        return portfolio.get('total', self.capital) if isinstance(portfolio, dict) else self.capital
        
    def load_previous_params(self):
        """Load previous optimization parameters if available"""
        try:
            params_file = Path("./optimization_results/live_trader-params.json")
            if params_file.exists():
                with open(params_file, 'r') as f:
                    data = json.load(f)
                    self.current_params = data.get('parameters', {})
                    print(f"Loaded previous parameters with fitness: {data.get('fitness', 'N/A')}")
            else:
                print("No previous optimization results found - starting fresh")
        except Exception as e:
            print(f"Error loading previous parameters: {e}")
            self.current_params = None
        
    def play_sound(self, sound_type: str = "default"):
        """Play system sound for notifications"""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            if sound_type == "buy":
                os.system("afplay /System/Library/Sounds/Glass.aiff")
            elif sound_type == "sell":
                os.system("afplay /System/Library/Sounds/Ping.aiff")
            else:
                os.system("afplay /System/Library/Sounds/Pop.aiff")
        elif system == "Linux":
            os.system("beep")
    
    def load_cached_data(self) -> pd.DataFrame:
        """Load cached bar data from CSV if exists"""
        if self.csv_filename.exists():
            try:
                df = pd.read_csv(self.csv_filename, parse_dates=['timestamp'])
                print(f"Loaded {len(df)} cached bars from {self.csv_filename.name}")
                return df
            except Exception as e:
                print(f"Error loading cached data: {e}")
                return pd.DataFrame()
        else:
            print(f"No cached data found. Will create new cache file: {self.csv_filename.name}")
            return pd.DataFrame()
    
    def save_bars_to_cache(self, new_bars: pd.DataFrame):
        """Append new bars to CSV cache"""
        if new_bars.empty:
            return
            
        # If cache exists, append only new bars
        if not self.cached_df.empty:
            # Get the last timestamp in cache
            last_cached_time = self.cached_df['timestamp'].max()
            
            # Filter only new bars
            new_bars = new_bars[new_bars['timestamp'] > last_cached_time]
            
            if new_bars.empty:
                return
        
        # Append to cache
        self.cached_df = pd.concat([self.cached_df, new_bars], ignore_index=True)
        
        # Remove any duplicates and sort
        self.cached_df = self.cached_df.drop_duplicates(subset=['timestamp'], keep='first')
        self.cached_df.sort_values('timestamp', inplace=True)
        
        # Save to CSV
        try:
            self.cached_df.to_csv(self.csv_filename, index=False)
            print(f"Cached {len(new_bars)} new bars. Total cached: {len(self.cached_df)}")
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def fetch_missing_bars(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch bars for a specific time range to fill gaps"""
        # Create timeframe object
        if self.timeframe_minutes < 60:
            timeframe = TimeFrame(self.timeframe_minutes, TimeFrameUnit.Minute)  # type: ignore
        elif self.timeframe_minutes == 60:
            timeframe = TimeFrame(1, TimeFrameUnit.Hour)  # type: ignore
        elif self.timeframe_minutes == 1440:
            timeframe = TimeFrame(1, TimeFrameUnit.Day)  # type: ignore
        else:
            timeframe = TimeFrame(self.timeframe_minutes, TimeFrameUnit.Minute)  # type: ignore
        
        # Fetch data
        if self.is_crypto:
            request = CryptoBarsRequest(
                symbol_or_symbols=self.symbol,
                start=start_time,
                end=end_time,
                timeframe=timeframe
            )
            bars = self.data_client.get_crypto_bars(request)  # type: ignore
        else:
            request = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                start=start_time,
                end=end_time,
                timeframe=timeframe
            )
            bars = self.data_client.get_stock_bars(request)  # type: ignore
            
        # Convert to DataFrame
        # The bars object is a BarSet, access it directly with symbol
        symbol_bars = bars[self.symbol]
        
        # If it's a list of bars, convert to dataframe manually
        if isinstance(symbol_bars, list):
            # Create DataFrame from list of bars
            data = []
            for bar in symbol_bars:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'trade_count': bar.trade_count,
                    'vwap': bar.vwap
                })
            df = pd.DataFrame(data)
        else:
            # Otherwise use the .df attribute
            df = symbol_bars.df
            df = df.reset_index()
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
        
        # For crypto, try to get latest bar if we're missing recent data
        # Be careful: latest bars might be "live" and change between calls
        if self.is_crypto and not df.empty:
            latest_fetched = df['timestamp'].max()
            expected_latest = end_time - timedelta(minutes=self.timeframe_minutes)
            
            if latest_fetched < expected_latest:
                try:
                    # Try to get the latest bar using the correct API
                    from alpaca.data.requests import CryptoLatestBarRequest
                    latest_request = CryptoLatestBarRequest(symbol_or_symbols=self.symbol)
                    latest_bars = self.data_client.get_crypto_latest_bar(latest_request)  # type: ignore
                    
                    if latest_bars and self.symbol in latest_bars:
                        bar = latest_bars[self.symbol]
                        
                        # Only add bars that are definitely closed (not current period)
                        current_period_start = datetime.now(timezone.utc).replace(
                            minute=(datetime.now(timezone.utc).minute // self.timeframe_minutes) * self.timeframe_minutes,
                            second=0, microsecond=0
                        )
                        
                        if bar.timestamp > latest_fetched and bar.timestamp < current_period_start:
                            df = pd.concat([df, pd.DataFrame([{
                                'timestamp': bar.timestamp,
                                'open': bar.open,
                                'high': bar.high,
                                'low': bar.low,
                                'close': bar.close,
                                'volume': bar.volume,
                                'trade_count': bar.trade_count,
                                'vwap': bar.vwap
                            }])], ignore_index=True)
                except Exception as e:
                    pass  # Silently handle errors in gap filling
        
        # Forward fill gaps in the data
        if not df.empty:
            df = impute_missing_bars(df, self.timeframe_minutes)
        
        return df
        
    def get_historical_data(self, lookback_bars: int = 200) -> pd.DataFrame:
        """Fetch historical data using cache when possible"""
        end = datetime.now(timezone.utc)
        
        # If cache is empty (first start or new instrument), fetch more history
        if self.cached_df.empty:
            print(f"First time seeing {self.symbol}, fetching {self.initial_history_bars} bars of history...")
            lookback_bars = max(lookback_bars, self.initial_history_bars)
        
        # Calculate desired start time
        total_minutes = int(lookback_bars * self.timeframe_minutes * 1.5)
        desired_start = end - timedelta(minutes=total_minutes)
        
        # Check if we have enough cached data
        if not self.cached_df.empty:
            # Find the time range we need to fetch
            earliest_cached = self.cached_df['timestamp'].min()
            latest_cached = self.cached_df['timestamp'].max()
            
            # Determine what data to fetch
            need_older_data = desired_start < earliest_cached and not self.historical_data_limit_reached
            need_newer_data = latest_cached < (end - timedelta(minutes=self.timeframe_minutes * 2))
            
            if need_older_data:
                # Fetch older data
                print(f"Fetching historical data before {earliest_cached}")
                older_bars = self.fetch_missing_bars(desired_start, earliest_cached)
                if not older_bars.empty:
                    print(f"Successfully fetched {len(older_bars)} older bars")
                    self.cached_df = pd.concat([older_bars, self.cached_df], ignore_index=True)
                    # Remove duplicates that might occur from overlapping fetches
                    self.cached_df = self.cached_df.drop_duplicates(subset=['timestamp'], keep='first')
                    self.cached_df.sort_values('timestamp', inplace=True)
                    self.save_bars_to_cache(older_bars)
                else:
                    print("No older bars returned - reached historical data limit")
                    self.historical_data_limit_reached = True
            
            if need_newer_data:
                # Fetch newer data (gap filling)
                newer_bars = self.fetch_missing_bars(latest_cached, end)
                if not newer_bars.empty:
                    self.save_bars_to_cache(newer_bars)
            
            # Return data from cache (always return what we have, even if less than requested)
            result = self.cached_df[self.cached_df['timestamp'] >= desired_start].copy()
            if len(result) > 0:
                # Show cache usage with dots for repeated fetches
                self.cache_fetch_count += 1
                if self.cache_fetch_count == 1:
                    print(f"Using {len(result)} cached bars (requested {lookback_bars})", end='', flush=True)
                    self.cache_message_printed = True
                else:
                    # Add a dot for each subsequent fetch
                    print(".", end='', flush=True)
                return result.tail(min(lookback_bars * 2, len(result)))  # Return what we have, up to requested * 2
            
            # If we have cache but no results in the time range, return recent data
            if not self.cached_df.empty:
                print(f"Time range too far back, returning most recent {min(lookback_bars, len(self.cached_df))} bars")
                return self.cached_df.tail(min(lookback_bars, len(self.cached_df)))
        
        # If no cache at all, fetch fresh data
        print(f"Initializing with {lookback_bars} bars of historical data")
        fresh_df = self.fetch_missing_bars(desired_start, end)
        
        # Save to cache
        if not fresh_df.empty:
            self.save_bars_to_cache(fresh_df)
        
        return fresh_df
    
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            pricing = self.alpaca_client.get_pricing(self.symbol)
            return pricing.bid_ask_midpoint
        except Exception as e:
            print(f"Error getting current price: {e}")
            bars = self.alpaca_client.get_hour_bars(hours=1, length=1, symbol=self.symbol)
            if bars:
                return float(bars[-1].close)
            raise
    
    def run_optimization_on_dataframe(self, df: pd.DataFrame) -> Dict:
        """Run optimization directly on DataFrame without file I/O"""
        from optimization import GeneticAlgorithm, create_enhanced_parameter_ranges, estimate_asset_volatility_scale
        from data_processing import prepare_data, create_dollar_bars
        
        # Prepare data
        df_prepared = prepare_data(df.copy())
        
        # Handle dollar bars
        use_dollar_bars = self.dollar_threshold not in [None, "none", "off", "disable", "false"]
        if use_dollar_bars:
            if self.dollar_threshold == "auto":
                from optimization import auto_detect_dollar_thresholds
                dollar_threshold = auto_detect_dollar_thresholds(df_prepared)
                print(f"Selected dollar threshold: ${dollar_threshold:,.0f}")
            else:
                try:
                    dollar_threshold = float(self.dollar_threshold)
                    print(f"Using specified dollar threshold: ${dollar_threshold:,.0f}")
                except (ValueError, TypeError):
                    from optimization import auto_detect_dollar_thresholds
                    dollar_threshold = auto_detect_dollar_thresholds(df_prepared)
                    print(f"Invalid threshold, using auto-detect: ${dollar_threshold:,.0f}")
            
            print(f"Creating dollar bars (threshold: ${dollar_threshold:,.0f})...")
            df_prepared = create_dollar_bars(df_prepared, dollar_threshold)
            print(f"Dollar bars created: {len(df_prepared)}")
        
        # Estimate volatility scale
        volatility_scale = estimate_asset_volatility_scale(df_prepared)
        
        # Create parameter ranges
        param_ranges = create_enhanced_parameter_ranges(volatility_scale)
        
        # Initialize genetic algorithm
        ga = GeneticAlgorithm(
            parameter_config=param_ranges,
            population_size=self.population,
            generations=self.generations,
            fitness_metric=self.fitness
        )
        
        # Seed with previous best parameters if available
        if self.current_params is not None:
            print("\nSeeding GA with previous best parameters...")
            # Add previous best to archive so it gets used in population creation
            ga.candidate_archive.insert(0, self.current_params.copy())
            
        # Pass archive of top candidates if available
        if self.ga_archive:
            ga.candidate_archive = self.ga_archive.copy()
            print(f"Loaded archive with {len(self.ga_archive)} top candidates")
        
        print("\nRunning optimization...")
        print()
        
        # Run optimization
        best_individual, fitness_history = ga.optimize(df_prepared, verbose=True)
        
        # Save the updated archive for next optimization
        self.ga_archive = ga.candidate_archive.copy()
        
        print(f"\nBest fitness: {best_individual.fitness:.4f}")
        print("\nBest parameters:")
        for param, value in best_individual.genes.items():
            print(f"  {param}: {value:.4f}")
        
        # Prepare results data
        result_data = {
            'parameters': best_individual.genes.copy(),
            'fitness': best_individual.fitness,
            'fitness_metric': self.fitness,
            'allow_shorts': self.allow_shorts,
            'data_file': "live_trader_dataframe",  # Indicate it was from DataFrame
            'dollar_threshold': dollar_threshold if use_dollar_bars else None,
            'population_size': self.population,
            'generations': self.generations,
            'parameter_ranges': {k: v.to_dict() for k, v in param_ranges.items()},
            'fitness_progression': {
                'initial': fitness_history[0] if fitness_history else best_individual.fitness,
                'final': best_individual.fitness,
                'improvement': best_individual.fitness - (fitness_history[0] if fitness_history else best_individual.fitness),
                'generations_to_converge': len(fitness_history)
            }
        }
        
        # Save results for compatibility with test function
        print("\nSaving optimization results...")
        results_dir = Path("./optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        params_file = results_dir / "live_trader-params.json"
        with open(params_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"   Optimized parameters saved to: {params_file}")
        
        return result_data

    def run_test_on_dataframe(self, df: pd.DataFrame, params_data: Dict) -> Dict:
        """Run test directly on DataFrame without file I/O"""
        from data_processing import prepare_data, create_dollar_bars
        from factors import calculate_mss, calculate_macd, calculate_rsi
        from strategy_enhanced import EnhancedTradingStrategy
        from performance import generate_performance_report
        
        print("\nHelios Strategy Testing")
        print("=" * 40)
        
        opt_params = params_data['parameters']
        allow_shorts = params_data.get('allow_shorts', False)
        dollar_threshold = params_data.get('dollar_threshold')
        
        print("Loaded strategy parameters:")
        for param, value in opt_params.items():
            print(f"  {param}: {value:.4f}")
        print(f"Original fitness: {params_data.get('fitness', 'N/A')}")
        print(f"Allow shorts: {allow_shorts}")
        print(f"Use dollar bars: {dollar_threshold is not None}")
        if dollar_threshold:
            print(f"Dollar threshold: ${dollar_threshold:,.0f}")
        print(f"Using cached DataFrame directly")
        
        # Prepare data
        df_prepared = prepare_data(df.copy())
        print(f"\nData loaded: {len(df_prepared)} bars")
        print(f"Date range: {df_prepared.index[0]} to {df_prepared.index[-1]}")
        
        # Handle dollar bars
        if dollar_threshold:
            print(f"\nCreating dollar bars (threshold: ${dollar_threshold:,.0f})...")
            df_prepared = create_dollar_bars(df_prepared, dollar_threshold)
            print(f"Dollar bars created: {len(df_prepared)}")
        else:
            print(f"\nUsing traditional candle data ({len(df_prepared)} bars)")
        
        # Calculate indicators
        print("\nCalculating indicators...")
        
        # Calculate MSS
        weights = {
            'trend': opt_params.get('weight_trend', 0.4),
            'volatility': opt_params.get('weight_volatility', 0.3),
            'exhaustion': opt_params.get('weight_exhaustion', 0.3),
        }
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        lookback = int(opt_params.get('lookback_int', 20))
        factors_df, regimes = calculate_mss(df_prepared, lookback, weights)
        
        # Add additional indicators
        macd_data = calculate_macd(df_prepared)
        factors_df["macd_hist"] = macd_data["histogram"]
        factors_df["rsi"] = calculate_rsi(df_prepared)
        
        # Merge data
        combined_df = pd.concat([df_prepared, factors_df], axis=1)
        
        print(f"\nRunning backtest with enhanced strategy...")
        
        # Create strategy
        strategy = EnhancedTradingStrategy(
            initial_capital=self.capital,
            max_position_fraction=opt_params.get("max_position_pct", 1.0),
            entry_step_size=opt_params.get("entry_step_size", 0.2),
            stop_loss_multiplier_strong=opt_params.get("stop_loss_multiplier_strong", 2.0),
            stop_loss_multiplier_weak=opt_params.get("stop_loss_multiplier_weak", 1.0),
            strong_bull_threshold=opt_params.get("strong_bull_threshold", 50.0),
            weak_bull_threshold=opt_params.get("weak_bull_threshold", 20.0),
            neutral_upper=opt_params.get("neutral_threshold_upper", 20.0),
            neutral_lower=opt_params.get("neutral_threshold_lower", -20.0),
            weak_bear_threshold=opt_params.get("weak_bear_threshold", -20.0),
            strong_bear_threshold=opt_params.get("strong_bear_threshold", -50.0),
            allow_shorts=allow_shorts,
        )
        
        # Run backtest
        results = strategy.run_backtest(combined_df, combined_df)
        trades_summary = strategy.get_trade_summary()
        
        # Check if results DataFrame has required columns
        if results.empty:
            print("Warning: Backtest returned empty results")
            return None
            
        if 'portfolio_value' not in results.columns:
            print(f"Warning: Backtest results missing 'portfolio_value' column")
            print(f"Available columns: {list(results.columns)}")
            return None
        
        # Generate performance report
        performance_report = generate_performance_report(
            results, trades_summary, self.capital
        )
        print("\n" + performance_report)
        
        return {
            'results': results,
            'trades_summary': trades_summary,
            'performance_report': performance_report,
            'strategy': strategy
        }

    def run_optimization_and_test(self, df: pd.DataFrame) -> Tuple[Dict, object]:
        """Run optimization and test using main.py functions directly"""
        print(f"\n{cyan(datetime.now().strftime('%H:%M:%S'))} - {yellow('Running optimization...')}", end='', flush=True)
        
        start_time = time.time()
        
        # Use rolling window with maximum number of bars
        total_bars = len(df)
        max_bars = min(self.max_optimization_bars, total_bars)
        df_opt = df.tail(max_bars).copy()
        
        # Ensure we have enough data
        min_bars = max(self.lookback_bars, 200)
        if len(df_opt) < min_bars:
            df_opt = df.copy()
        
        bars_used = len(df_opt)
        
        try:
            # Run optimization directly on DataFrame instead of saving to file
            params_data = self.run_optimization_on_dataframe(df_opt)
            
            if params_data is None:
                raise Exception("Optimization failed")
            
            opt_params = params_data['parameters']
            
            # Parameters file is saved for test command compatibility
            params_file = Path("./optimization_results/live_trader-params.json")
            
            # Run test directly on DataFrame
            test_result = self.run_test_on_dataframe(df_opt, params_data)
            
            if test_result is None:
                raise Exception("Test failed")
            
            elapsed = time.time() - start_time
            # Print newline if we've been showing cache dots
            if self.cache_message_printed:
                print()  # New line to clear the dots
                self.cache_message_printed = False
                self.cache_fetch_count = 0
            print(f" {green('Done!')} ({cyan(f'{elapsed:.1f}s')})")
            fitness_val = params_data.get("fitness", 0)
            print(f"  {white('Best fitness:')} {cyan(f'{fitness_val:.4f}')} | {white('Window:')} {cyan(f'{bars_used} bars')}")
            
            # Record optimization result
            self.optimization_history.append({
                'timestamp': datetime.now(timezone.utc),
                'fitness': params_data.get('fitness', 0),
                'params': opt_params.copy(),
                'elapsed_seconds': elapsed,
                'bars_used': bars_used,
                'params_file': str(params_file)
            })
            
            return opt_params, params_data
            
        except Exception as e:
            print(f" {red('Failed:')} {e}")
            # Return default parameters on failure
            default_params = {
                'strong_bull_threshold': 50.0,
                'weak_bull_threshold': 20.0,
                'neutral_threshold_upper': 20.0,
                'neutral_threshold_lower': -20.0,
                'weak_bear_threshold': -20.0,
                'strong_bear_threshold': -50.0,
                'max_position_pct': 1.0,
                'entry_step_size': 0.2,
                'stop_loss_multiplier_strong': 2.0,
                'stop_loss_multiplier_weak': 1.0,
                'lookback_int': 20,
                'weight_trend': 0.4,
                'weight_volatility': 0.3,
                'weight_exhaustion': 0.3
            }
            return default_params, {'parameters': default_params}
    
    def get_signals_from_strategy(self, df: pd.DataFrame, params: Dict) -> Tuple[bool, bool, str, Dict]:
        """Generate signals using the same logic as main.py test"""
        if len(df) < 50:
            return False, False, "Insufficient data", {}
        
        try:
            # Process data exactly like main.py test
            df_prepared = prepare_data(df.copy())
            
            # Handle dollar bars if enabled
            dollar_threshold = params.get('dollar_threshold')
            use_dollar_bars = dollar_threshold is not None
            
            if use_dollar_bars and dollar_threshold:
                df_prepared = create_dollar_bars(df_prepared, dollar_threshold)
                print(f"Dollar bars created: {len(df_prepared)} from {len(df)} time bars")
                if len(df_prepared) < 50:
                    return False, False, "Insufficient dollar bars", {}
            
            # Calculate factors with optimized parameters
            weights = {
                'trend': params.get('weight_trend', 0.4),
                'volatility': params.get('weight_volatility', 0.3),
                'exhaustion': params.get('weight_exhaustion', 0.3),
            }
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
            
            lookback = int(params.get('lookback_int', 20))
            factors_df, regimes = calculate_mss(df_prepared, lookback, weights)
            
            # Add additional indicators
            macd_data = calculate_macd(df_prepared)
            factors_df['macd_hist'] = macd_data['histogram']
            factors_df['rsi'] = calculate_rsi(df_prepared)
            
            # Merge data
            combined_df = pd.concat([df_prepared, factors_df], axis=1)
            
            # Get current market state
            latest = combined_df.iloc[-1]
            mss_score = latest.get('mss', 0)
            
            # Debug: Check if data processing worked
            if len(combined_df) == 0:
                return False, False, "Empty combined dataframe", {}
            
            # Check if mss calculation succeeded
            if 'mss' not in combined_df.columns:
                print(f"Warning: MSS not calculated. Available columns: {list(combined_df.columns)}")
                return False, False, "MSS calculation failed", {}
            
            # Determine regime
            if mss_score >= params.get('strong_bull_threshold', 50):
                regime = 'Strong Bull'
            elif mss_score >= params.get('weak_bull_threshold', 20):
                regime = 'Weak Bull'
            elif mss_score >= params.get('neutral_threshold_upper', 20):
                regime = 'Neutral Upper'
            elif mss_score >= params.get('neutral_threshold_lower', -20):
                regime = 'Neutral Lower'
            elif mss_score >= params.get('weak_bear_threshold', -20):
                regime = 'Weak Bear'
            else:
                regime = 'Strong Bear'
            
            # Create strategy instance
            strategy = EnhancedTradingStrategy(
                initial_capital=self.capital,
                max_position_fraction=params.get('max_position_pct', 1.0),
                entry_step_size=params.get('entry_step_size', 0.2),
                stop_loss_multiplier_strong=params.get('stop_loss_multiplier_strong', 2.0),
                stop_loss_multiplier_weak=params.get('stop_loss_multiplier_weak', 1.0),
                strong_bull_threshold=params.get('strong_bull_threshold', 50.0),
                weak_bull_threshold=params.get('weak_bull_threshold', 20.0),
                neutral_upper=params.get('neutral_threshold_upper', 20.0),
                neutral_lower=params.get('neutral_threshold_lower', -20.0),
                weak_bear_threshold=params.get('weak_bear_threshold', -20.0),
                strong_bear_threshold=params.get('strong_bear_threshold', -50.0),
                allow_shorts=self.allow_shorts,
            )
            
            # Get target position fraction for current regime
            target_position_fraction = strategy.get_target_position_fraction(regime, mss_score)
            
            # Calculate current position value
            current_price = latest['close']
            current_position_value = self.current_position_value(current_price)
            portfolio_value = self.get_portfolio_value(current_price)
            current_position_fraction = current_position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Determine signals based on target vs current position
            buy_signal = False
            sell_signal = False
            reason = ""
            
            position_diff = target_position_fraction - current_position_fraction
            
            if abs(position_diff) > 0.05:  # 5% threshold for position changes
                if position_diff > 0:
                    buy_signal = True
                    reason = f"{regime}: Target {target_position_fraction:.1%} vs Current {current_position_fraction:.1%}"
                else:
                    sell_signal = True
                    reason = f"{regime}: Target {target_position_fraction:.1%} vs Current {current_position_fraction:.1%}"
            
            market_info = {
                'mss_score': mss_score,
                'regime': regime,
                'target_position_fraction': target_position_fraction,
                'current_position_fraction': current_position_fraction,
                'atr': latest.get('atr', 0),
                'rsi': latest.get('rsi', 50),
                'macd_hist': latest.get('macd_hist', 0)
            }
            
            return buy_signal, sell_signal, reason, market_info
            
        except Exception as e:
            print(f"Error in signal generation: {e}")
            return False, False, f"Error: {e}", {}
    
    def check_new_bar(self, df: pd.DataFrame) -> bool:
        """Check if a new bar has arrived"""
        if df.empty:
            return False
            
        latest_bar_time = df.iloc[-1]['timestamp']
        
        if self.last_bar_time is None:
            self.last_bar_time = latest_bar_time
            return True
        
        if latest_bar_time > self.last_bar_time:
            self.last_bar_time = latest_bar_time
            return True
            
        return False
    
    def current_position_value(self, current_price: float) -> float:
        """Get current position value from tracker"""
        return self.current_position_size * current_price
    
    
    def execute_trade(self, side: OrderSide, quantity: float, price: float, reason: str):
        """Execute a paper trade using the comprehensive ledger system"""
        try:
            # Convert OrderSide to string
            side_str = side.value.lower() if hasattr(side, 'value') else str(side).lower()
            
            # Record trade in tracker with slippage and fees
            trade_entry = self.tracker.record_trade(
                side=side_str,
                quantity=quantity,
                market_price=price,
                notes=reason
            )
            
            # Play appropriate sound
            self.play_sound("buy" if side_str == "buy" else "sell")
            
            # Get portfolio metrics
            portfolio = self.tracker.get_portfolio_value(price)
            
            if side_str == "buy":
                print(f"\n{green('═' * 60)}")
                print(f"{green('🔔 BUY SIGNAL')} - {white(str(datetime.now()))}")
                print(f"{green('═' * 60)}")
                print(f"{white('Symbol:')} {bold(self.symbol)}")
                print(f"{white('Reason:')} {green(reason)}")
                print(f"{white('Quantity:')} {green(f'{quantity:.6f}')}")
                print(f"{white('Market Price:')} ${green(f'{price:.4f}')}")
                print(f"{white('Execution Price:')} ${green(f'{trade_entry["price"]:.4f}')} (slippage: {trade_entry['slippage_bps']:.1f}bps)")
                print(f"{white('Trade Value:')} ${green(f'{trade_entry["trade_value"]:.2f}')}")
                print(f"{white('Fees:')} ${cyan(f'{trade_entry["fee_total"]:.2f}')}")
                print(f"{white('Net Amount:')} ${red(f'{abs(trade_entry["net_amount"]):.2f}')} (out)")
                print(f"{white('Cash Remaining:')} ${cyan(f'{portfolio["cash"]:.2f}')}")
                print(f"{white('Position:')} {cyan(f'{portfolio["position_size"]:.6f}')} {self.symbol}")
            else:
                profit_color = green if trade_entry['realized_pnl'] >= 0 else red
                print(f"\n{red('═' * 60)}")
                print(f"{red('🔔 SELL SIGNAL')} - {white(str(datetime.now()))}")
                print(f"{red('═' * 60)}")
                print(f"{white('Symbol:')} {bold(self.symbol)}")
                print(f"{white('Reason:')} {red(reason)}")
                print(f"{white('Quantity:')} {red(f'{quantity:.6f}')}")
                print(f"{white('Market Price:')} ${red(f'{price:.4f}')}")
                print(f"{white('Execution Price:')} ${red(f'{trade_entry["price"]:.4f}')} (slippage: {trade_entry['slippage_bps']:.1f}bps)")
                print(f"{white('Trade Value:')} ${red(f'{trade_entry["trade_value"]:.2f}')}")
                print(f"{white('Fees:')} ${cyan(f'{trade_entry["fee_total"]:.2f}')}")
                print(f"{white('Net Amount:')} ${green(f'{trade_entry["net_amount"]:.2f}')} (in)")
                print(f"{white('Realized P&L:')} ${profit_color(f'{trade_entry["realized_pnl"]:.2f}')}")
                print(f"{white('Cash:')} ${cyan(f'{portfolio["cash"]:.2f}')}")
                print(f"{white('Position:')} {cyan(f'{portfolio["position_size"]:.6f}')} {self.symbol}")
            
            # Show portfolio summary
            print(f"\n{yellow('📊 PORTFOLIO SUMMARY:')}")
            print(f"   {white('Total Value:')} ${cyan(f'{portfolio["total_value"]:.2f}')}")
            print(f"   {white('Total P&L:')} ${(green if portfolio['total_pnl'] >= 0 else red)(f'{portfolio["total_pnl"]:.2f}')}")
            print(f"   {white('Total Return:')} {(green if portfolio['total_return_pct'] >= 0 else red)(f'{portfolio["total_return_pct"]:.2f}%')}")
            
            if self.current_params:
                print(f"\n{yellow('🎯 STRATEGY PARAMETERS:')}")
                for key, value in self.current_params.items():
                    if isinstance(value, float):
                        print(f"   {white(key)}: {cyan(f'{value:.4f}')}")
                    else:
                        print(f"   {white(key)}: {cyan(str(value))}")
            
            print(f"{green('═' * 60) if side_str == 'buy' else red('═' * 60)}\n")
            
            # Record trade for tracking
            trade_record = {
                'timestamp': datetime.now(timezone.utc),
                'side': side_str.upper(),
                'quantity': quantity,
                'price': trade_entry['price'],
                'market_price': price,
                'slippage_bps': trade_entry['slippage_bps'],
                'fees': trade_entry['fee_total'],
                'reason': reason,
                'portfolio_value': portfolio['total_value']
            }
            self.trades.append(trade_record)
            
        except Exception as e:
            print(f"{red('TRADE EXECUTION FAILED:')} {e}")
            print(f"Side: {side}, Quantity: {quantity}, Price: {price}")
            raise
            
        # Sell logic now handled in the main execute_trade method above
            
            print(f"\n{red('═' * 60)}")
            print(f"{red('🔔 SELL SIGNAL')} - {white(str(datetime.now()))}")
            print(f"{red('═' * 60)}")
            print(f"{white('Symbol:')} {bold(self.symbol)}")
            print(f"{white('Reason:')} {red(reason)}")
            print(f"{white('Quantity:')} {red(f'{self.position.units:.4f}')}")
            print(f"{white('Price:')} ${red(f'{price:.2f}')}")
            print(f"{white('Proceeds:')} ${cyan(f'{proceeds:.2f}')}")
            print(f"{white('Entry Price:')} ${white(f'{self.position.avg_price:.2f}')}")
            print(f"{white('Profit:')} {profit_color(f'${profit:.2f} ({profit_pct:.2f}%)')}")
            print(f"{white('Cash After Sale:')} ${cyan(f'{self.cash:.2f}')}")
            print(f"\n{yellow('📋 MANUAL ORDER DETAILS:')}")
            print(f"   {white('Order Type:')} {red('MARKET SELL')}")
            print(f"   {white('Symbol:')} {bold(self.symbol)}")
            print(f"   {white('Quantity:')} {red(f'{self.position.units:.4f}')}")
            print(f"   {white('Estimated Price:')} ${red(f'{price:.2f}')}")
            print(f"{red('═' * 60)}\n")
            
            # Record trade
            self.trades.append({
                'entry_time': self.position.entry_time,
                'exit_time': datetime.now(timezone.utc),
                'entry_price': self.position.avg_price,
                'exit_price': price,
                'quantity': self.position.units,
                'profit': profit,
                'profit_pct': profit_pct,
                'exit_reason': reason
            })
            
            # Reset position
            self.position = Position()
    
    def update_equity_curve(self, current_price: float):
        """Update equity tracking"""
        position_value = self.position.units * current_price if self.position.is_open else 0
        total_equity = self.cash + position_value
        self.equity_curve.append(total_equity)
    
    def print_status(self, current_price: float, df: pd.DataFrame, market_info: Dict | None = None):
        """Print current trading status with colors"""
        position_value = self.position.units * current_price if self.position.is_open else 0
        total_equity = self.cash + position_value
        
        # Calculate returns
        total_return = ((total_equity - self.capital) / self.capital * 100)
        return_color = green if total_return >= 0 else red
        
        # Get market info from the strategy
        if market_info:
            regime = market_info.get('regime', 'Unknown')
            mss_score = market_info.get('mss_score', 0)
            target_fraction = market_info.get('target_position_fraction', 0)
            current_fraction = market_info.get('current_position_fraction', 0)
            atr = market_info.get('atr', 0)
            rsi = market_info.get('rsi', 50)
            macd_hist = market_info.get('macd_hist', 0)
        else:
            regime = "Unknown"
            mss_score = 0
            target_fraction = 0
            current_fraction = 0
            atr = 0
            rsi = 50
            macd_hist = 0
        
        # Determine regime color
        if 'Bull' in regime:
            regime_color = green
        elif 'Bear' in regime:
            regime_color = red
        else:
            regime_color = yellow
        
        # Header
        print(f"\n{cyan('═' * 80)}")
        
        # Time and Symbol
        print(f"{white('Time:')} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
              f"{white('Bar:')} {self.last_bar_time}")
        print(f"{white('Symbol:')} {bold(self.symbol)} | "
              f"{white('Price:')} ${cyan(f'{current_price:.2f}')} | "
              f"{white('Regime:')} {regime_color(bold(regime))}")
        
        # Market Indicators
        print(f"{white('MSS Score:')} {cyan(f'{mss_score:+.1f}')} | "
              f"{white('RSI:')} {cyan(f'{rsi:.1f}')} | "
              f"{white('ATR:')} {cyan(f'{atr:.4f}')} | "
              f"{white('MACD Hist:')} {cyan(f'{macd_hist:+.4f}')}")
        
        # Position Targets
        print(f"{white('Target Position:')} {cyan(f'{target_fraction:.1%}')} | "
              f"{white('Current Position:')} {cyan(f'{current_fraction:.1%}')} | "
              f"{white('Difference:')} {cyan(f'{(target_fraction - current_fraction):+.1%}')}")
        
        # Account Status
        print(f"\n{yellow('📊 SIMULATED ACCOUNT')}")
        print(f"  {white('Equity:')} ${bold(f'{total_equity:.2f}')} | "
              f"{white('Cash:')} ${f'{self.cash:.2f}'} | "
              f"{white('Return:')} {return_color(bold(f'{total_return:+.2f}%'))}")
        
        # Position Info
        if self.position.is_open:
            unrealized_pnl = position_value - self.position.cost_basis
            unrealized_pct = (unrealized_pnl / self.position.cost_basis) * 100
            pnl_color = green if unrealized_pnl >= 0 else red
            
            print(f"\n{yellow('📈 POSITION')}")
            print(f"  {white('Size:')} {self.position.units:.4f} @ ${self.position.avg_price:.2f}")
            print(f"  {white('Value:')} ${position_value:.2f} | "
                  f"{white('P&L:')} {pnl_color(f'${unrealized_pnl:+.2f} ({unrealized_pct:+.2f}%)')}")
        else:
            print(f"\n{yellow('📈 POSITION:')} {white('None')}")
            
        # Trading Stats - only count completed trades (those with 'profit' key)
        completed_trades = [t for t in self.trades if 'profit' in t]
        if completed_trades:
            total_profit = sum(t['profit'] for t in completed_trades)
            win_rate = sum(1 for t in completed_trades if t['profit'] > 0) / len(completed_trades) * 100
            profit_color = green if total_profit >= 0 else red
            
            print(f"\n{yellow('📊 TRADING STATS')}")
            print(f"  {white('Trades:')} {len(completed_trades)} completed | "
                  f"{white('Win Rate:')} {green(f'{win_rate:.1f}%') if win_rate >= 50 else red(f'{win_rate:.1f}%')} | "
                  f"{white('Total P&L:')} {profit_color(f'${total_profit:+.2f}')}")
            
        # Optimization Stats
        if len(self.optimization_history) > 1:
            recent_opt = self.optimization_history[-5:]
            avg_fitness = sum(opt['fitness'] for opt in recent_opt) / len(recent_opt)
            avg_time = sum(opt['elapsed_seconds'] for opt in recent_opt) / len(recent_opt)
            
            print(f"\n{yellow('⚙️  OPTIMIZATION')}")
            print(f"  {white('Avg Fitness:')} {cyan(f'{avg_fitness:.4f}')} | "
                  f"{white('Avg Time:')} {cyan(f'{avg_time:.1f}s')}")
    
    def run(self):
        """Main trading loop - bar driven"""
        print(f"\n{cyan('═' * 60)}")
        print(f"{yellow('📈 LIVE PAPER TRADING SYSTEM')}")
        print(f"{cyan('═' * 60)}")
        print(f"{white('Symbol:')} {bold(self.symbol)}")
        print(f"{white('Timeframe:')} {cyan(f'{self.timeframe_minutes} minute bars')}")
        print(f"{white('GA Settings:')} Population={cyan(str(self.population))}, Generations={cyan(str(self.generations))}")
        print(f"{white('Walk-Forward:')} {cyan(f'{self.max_optimization_bars} bars')} rolling window")
        print(f"{white('Cache file:')} {cyan(self.csv_filename.name)}")
        if not self.cached_df.empty:
            print(f"{white('Cached data:')} {green(f'{len(self.cached_df)} bars')} from {cyan(str(self.cached_df['timestamp'].min()))} to {cyan(str(self.cached_df['timestamp'].max()))}")
        print(f"\n{yellow('Waiting for first bar...')}")
        print(f"{white('Press')} {red('Ctrl+C')} {white('to stop')}")
        print(f"{cyan('═' * 60)}")
        
        # Track bar timing
        next_bar_time = None
        
        while True:
            try:
                # Get current time
                now = datetime.now(timezone.utc)
                
                # Calculate when next bar should close
                if next_bar_time is None or now >= next_bar_time:
                    # Calculate next bar close time
                    minutes_since_midnight = now.hour * 60 + now.minute
                    bars_since_midnight = minutes_since_midnight // self.timeframe_minutes
                    next_bar_minutes = (bars_since_midnight + 1) * self.timeframe_minutes
                    
                    next_bar_time = now.replace(
                        hour=next_bar_minutes // 60,
                        minute=next_bar_minutes % 60,
                        second=0,
                        microsecond=0
                    )
                    
                    # Handle day rollover
                    if next_bar_time <= now:
                        next_bar_time += timedelta(days=1)
                
                # Fetch latest data
                df = self.get_historical_data(lookback_bars=max(self.max_optimization_bars, 250))
                
                # Check if new bar arrived
                if self.check_new_bar(df):
                    print(f"\n{blue('─' * 80)}")
                    print(f"{blue('🕐')} {white(f'New {self.timeframe_minutes}-min bar at')} {cyan(str(self.last_bar_time))}")
                    
                    # Initialize position if not exists
                    if not hasattr(self, 'position'):
                        self.position = Position()
                    if not hasattr(self, 'cash'):
                        self.cash = self.capital
                    
                    # Run optimization and test using main.py functions
                    self.current_params, params_data = self.run_optimization_and_test(df)
                    
                    # Generate signals using the same logic as main.py test
                    buy_signal, sell_signal, reason, market_info = self.get_signals_from_strategy(df, self.current_params)
                    
                    # Get current price
                    current_price = self.get_current_price()
                    
                    # Execute trades based on signals
                    if buy_signal:
                        # Calculate position size based on target position fraction
                        target_fraction = market_info.get('target_position_fraction', 0)
                        portfolio_value = self.get_portfolio_value(current_price)
                        target_position_value = portfolio_value * target_fraction
                        current_position_value = self.current_position_value(current_price)
                        
                        position_change = target_position_value - current_position_value
                        
                        if position_change > 0:
                            quantity = position_change / current_price
                            
                            if self.is_crypto:
                                quantity = round(quantity, 4)
                            else:
                                quantity = int(quantity)
                            
                            if quantity > 0 and quantity * current_price <= self.cash:
                                self.execute_trade(OrderSide.BUY, quantity, current_price, reason)
                    
                    elif sell_signal and self.position.is_open:
                        # Calculate how much to sell based on target position fraction
                        target_fraction = market_info.get('target_position_fraction', 0)
                        portfolio_value = self.get_portfolio_value(current_price)
                        target_position_value = portfolio_value * target_fraction
                        current_position_value = self.current_position_value(current_price)
                        
                        position_change = current_position_value - target_position_value
                        
                        if position_change > 0:
                            quantity = min(position_change / current_price, self.position.units)
                            
                            if self.is_crypto:
                                quantity = round(quantity, 4)
                            else:
                                quantity = int(quantity)
                            
                            if quantity > 0:
                                self.execute_trade(OrderSide.SELL, quantity, current_price, reason)
                    
                    # Update tracking
                    self.update_equity_curve(current_price)
                    self.print_status(current_price, df, market_info)
                    
                    # Show time to next bar
                    time_to_next = (next_bar_time - datetime.now(timezone.utc)).total_seconds()
                    print(f"\nNext bar in {int(time_to_next)}s at {next_bar_time.strftime('%H:%M:%S')}")
                
                # Wait before checking again
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print(f"\n\n{red('⏹  Stopping live trader...')}")
                # Close any open positions
                if self.position.is_open:
                    print(f"{yellow('Closing open position...')}")
                    current_price = self.get_current_price()
                    self.execute_trade(OrderSide.SELL, self.position.units, current_price, "Manual close")
                break
                
            except Exception as e:
                print(f"\n{red('❌ Error in trading loop:')} {white(str(e))}")
                import traceback
                traceback.print_exc()
                time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(description='Live Paper Trading with GA and CSV Caching')
    parser.add_argument('symbol', help='Trading symbol (e.g., MSFT, BTCUSD)')
    parser.add_argument('--api-key', required=True, help='Alpaca API key')
    parser.add_argument('--secret-key', required=True, help='Alpaca secret key')
    parser.add_argument('--crypto', action='store_true', help='Trade cryptocurrency')
    parser.add_argument('--timeframe', type=int, default=1, help='Bar timeframe in minutes (default: 1)')
    
    # GA optimization parameters
    parser.add_argument('--population', type=int, default=50, help='GA population size (default: 50)')
    parser.add_argument('--generations', type=int, default=20, help='GA generations (default: 20)')
    parser.add_argument('--lookback', type=int, default=200, help='Lookback bars for optimization (default: 200)')
    parser.add_argument('--initial-bars', type=int, default=1000, help='Initial historical bars to fetch (default: 1000)')
    parser.add_argument('--max-opt-bars', type=int, default=2000, help='Max bars for walk-forward optimization window (default: 2000)')
    parser.add_argument('--dollar-threshold', default="auto", help='Dollar volume threshold for bars (default: auto-detect, number, or "none" to disable)')
    parser.add_argument('--fitness', choices=['sortino', 'calmar'], default='sortino', help='Fitness metric to optimize (default: sortino)')
    parser.add_argument('--allow-shorts', action='store_true', help='Allow short positions (disabled by default)')
    
    # Trading parameters
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    parser.add_argument('--check-interval', type=int, default=5, help='Check interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Create and run trader
    trader = LiveTrader(
        api_key=args.api_key,
        secret_key=args.secret_key,
        symbol=args.symbol,
        is_crypto=args.crypto,
        timeframe_minutes=args.timeframe,
        population=args.population,
        generations=args.generations,
        lookback_bars=args.lookback,
        initial_history_bars=args.initial_bars,
        capital=args.capital,
        check_interval=args.check_interval,
        max_optimization_bars=args.max_opt_bars,
        dollar_threshold=args.dollar_threshold,
        fitness=args.fitness,
        allow_shorts=args.allow_shorts
    )
    
    try:
        trader.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
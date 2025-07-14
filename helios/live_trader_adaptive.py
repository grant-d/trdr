"""
Adaptive Live Paper Trading System
Runs GA optimization on every new bar for dynamic strategy adjustment
"""

import time
import argparse
import sys
import os
import platform
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import OrderSide

from alpaca_client import AlpacaClient
from data_processing import prepare_data
from optimization import GeneticAlgorithm
from strategy_enhanced import EnhancedTradingStrategy, Position


class AdaptiveLiveTrader:
    """Adaptive live paper trading with continuous optimization"""
    
    def __init__(self, api_key: str, secret_key: str, symbol: str, is_crypto: bool = False, timeframe_minutes: int = 1):
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
        
        # Simulated account state
        self.initial_capital = 100000.0
        self.cash = self.initial_capital
        self.position = Position()  # Use the Position class from strategy
        
        # Strategy and optimization
        self.strategy = None  # Will be created with optimized params
        self.current_params = None
        self.last_bar_time = None
        
        # GA settings for fast optimization
        self.ga_population = 20  # Small population for speed
        self.ga_generations = 10  # Few generations
        self.ga_lookback_bars = 200  # Optimize on recent history
        self.initial_history_bars = 1000  # Fetch this many bars on first start
        
        # Performance tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.initial_capital]
        self.optimization_history: List[Dict] = []
        
        # CSV caching setup
        self.cache_dir = Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Replace slash with underscore for filename
        safe_symbol = symbol.replace("/", "_")
        self.csv_filename = self.cache_dir / f"{safe_symbol}_{timeframe_minutes}min.csv"
        self.cached_df = self.load_cached_data()
        
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
            need_older_data = desired_start < earliest_cached
            need_newer_data = latest_cached < (end - timedelta(minutes=self.timeframe_minutes * 2))
            
            if need_older_data:
                # Fetch older data
                print(f"Fetching historical data before {earliest_cached}")
                older_bars = self.fetch_missing_bars(desired_start, earliest_cached)
                if not older_bars.empty:
                    self.cached_df = pd.concat([older_bars, self.cached_df], ignore_index=True)
                    self.cached_df.sort_values('timestamp', inplace=True)
                    self.save_bars_to_cache(older_bars)
            
            if need_newer_data:
                # Fetch newer data (gap filling)
                print(f"Fetching missing bars after {latest_cached}")
                newer_bars = self.fetch_missing_bars(latest_cached, end)
                if not newer_bars.empty:
                    self.save_bars_to_cache(newer_bars)
            
            # Return data from cache
            result = self.cached_df[self.cached_df['timestamp'] >= desired_start].copy()
            if len(result) >= lookback_bars:
                return result.tail(lookback_bars * 2)  # Return extra for safety
        
        # If no cache or insufficient data, fetch fresh
        if self.cached_df.empty:
            print(f"Initializing with {lookback_bars} bars of historical data")
        else:
            print(f"Insufficient cached data, fetching {lookback_bars} bars")
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
    
    def run_optimization(self, df: pd.DataFrame) -> Dict:
        """Run fast GA optimization on recent data"""
        print(f"\n{datetime.now().strftime('%H:%M:%S')} - Running optimization...", end='', flush=True)
        
        start_time = time.time()
        
        # Prepare data with indicators
        df_prepared = prepare_data(df, lookback=20)
        
        # Use only recent bars for optimization
        df_opt = df_prepared.tail(self.ga_lookback_bars).copy()
        
        # Define parameter ranges for GA
        parameter_config = {
            'entry_z_long': (0.3, 1.5),
            'exit_z_long': (-0.5, 0.5),
            'entry_z_short': (-1.5, -0.3),
            'exit_z_short': (-0.5, 0.5),
            'trailing_stop_pct': (1.0, 3.0),
            'max_position_fraction': (0.8, 1.0),
            'stop_loss_pct': (0.5, 2.0)
        }
        
        # Create GA with small population for speed
        ga = GeneticAlgorithm(
            parameter_config=parameter_config,
            population_size=self.ga_population,
            generations=self.ga_generations,
            mutation_rate=0.3,  # Higher mutation for diversity
            crossover_rate=0.8,
            elitism_rate=0.1
        )
        
        try:
            # Run optimization (returns tuple of best_individual and fitness_history)
            result = ga.optimize(df_opt, verbose=False)
            if isinstance(result, tuple):
                best_individual, _ = result
            else:
                best_individual = result
            
            elapsed = time.time() - start_time
            print(f" Done! ({elapsed:.1f}s)")
            print(f"  Best fitness: {best_individual.fitness:.4f}")
            
            # Get parameters from Individual's genes
            best_params = best_individual.genes.copy()
            
            # Record optimization result
            self.optimization_history.append({
                'timestamp': datetime.now(timezone.utc),
                'fitness': best_individual.fitness,
                'params': best_params.copy(),
                'elapsed_seconds': elapsed
            })
            
            return best_params
            
        except Exception as e:
            print(f" Failed: {e}")
            # Return default parameters on failure
            return {
                'entry_z_long': 0.5,
                'exit_z_long': -0.3,
                'entry_z_short': -0.5,
                'exit_z_short': 0.3,
                'trailing_stop_pct': 2.0,
                'max_position_fraction': 1.0,
                'stop_loss_pct': 1.0
            }
    
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
    
    def check_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, str]:
        """Check for buy/sell signals using current strategy"""
        if self.strategy is None or len(df) < 50:
            return False, False, "No strategy"
        
        # Prepare data
        df_prepared = prepare_data(df, lookback=20)
        
        # Get latest bar
        current_idx = len(df_prepared) - 1
        current_bar = df_prepared.iloc[current_idx]
        
        # Initialize signal tracking
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # Check position status
        has_position = self.position.is_open
        
        if not has_position:
            # Check buy conditions using z-score
            z_score = current_bar.get('z_score', 0)
            entry_z = self.current_params.get('entry_z_long', 0.5) if self.current_params else 0.5
            if z_score < -entry_z:
                buy_signal = True
                reason = f"Buy: Z-score {z_score:.2f} < -{entry_z:.2f}"
        else:
            # Check sell conditions
            current_price = current_bar['close']
            z_score = current_bar.get('z_score', 0)
            
            # Calculate P&L
            if self.position.units > 0:
                pnl_pct = (current_price - self.position.avg_price) / self.position.avg_price * 100
            else:
                pnl_pct = 0
            
            # Exit conditions
            stop_loss_pct = self.current_params.get('stop_loss_pct', 1.0) if self.current_params else 1.0
            trailing_stop_pct = self.current_params.get('trailing_stop_pct', 2.0) if self.current_params else 2.0
            exit_z = self.current_params.get('exit_z_long', -0.3) if self.current_params else -0.3
            
            if pnl_pct <= -stop_loss_pct:
                sell_signal = True
                reason = f"Stop Loss: {pnl_pct:.1f}% <= -{stop_loss_pct:.1f}%"
            elif z_score > exit_z:
                sell_signal = True
                reason = f"Exit Signal: Z-score {z_score:.2f} > {exit_z:.2f}"
            elif pnl_pct >= trailing_stop_pct * 2:  # Take profit at 2x trailing stop
                sell_signal = True
                reason = f"Take Profit: {pnl_pct:.1f}% >= {trailing_stop_pct * 2:.1f}%"
        
        return buy_signal, sell_signal, reason
    
    def execute_trade(self, side: OrderSide, quantity: float, price: float, reason: str):
        """Execute a paper trade with detailed logging"""
        if side == OrderSide.BUY:
            # Play buy sound
            self.play_sound("buy")
            
            # Calculate cost
            cost = quantity * price
            if cost > self.cash:
                quantity = self.cash / price
                cost = quantity * price
            
            # Update position and cash
            self.cash -= cost
            self.position.units = quantity
            self.position.cost_basis = cost
            self.position.entry_price = price
            self.position.entry_time = pd.Timestamp.now(tz='UTC')
            
            print(f"\n{'='*60}")
            print(f"🔔 BUY SIGNAL - {datetime.now()}")
            print(f"{'='*60}")
            print(f"Symbol: {self.symbol}")
            print(f"Reason: {reason}")
            print(f"Quantity: {quantity:.4f}")
            print(f"Price: ${price:.2f}")
            print(f"Total Cost: ${cost:.2f}")
            print(f"Cash Remaining: ${self.cash:.2f}")
            print(f"\n📊 OPTIMIZED PARAMETERS:")
            if self.current_params:
                for key, value in self.current_params.items():
                    print(f"   {key}: {value}")
            print(f"\n📋 MANUAL ORDER DETAILS:")
            print(f"   Order Type: MARKET BUY")
            print(f"   Symbol: {self.symbol}")
            print(f"   Quantity: {quantity:.4f}")
            print(f"   Estimated Price: ${price:.2f}")
            print(f"{'='*60}\n")
            
        else:  # SELL
            # Play sell sound
            self.play_sound("sell")
            
            # Calculate proceeds and profit
            proceeds = self.position.units * price
            self.cash += proceeds
            
            profit = proceeds - self.position.cost_basis
            profit_pct = (profit / self.position.cost_basis) * 100
            
            print(f"\n{'='*60}")
            print(f"🔔 SELL SIGNAL - {datetime.now()}")
            print(f"{'='*60}")
            print(f"Symbol: {self.symbol}")
            print(f"Reason: {reason}")
            print(f"Quantity: {self.position.units:.4f}")
            print(f"Price: ${price:.2f}")
            print(f"Proceeds: ${proceeds:.2f}")
            print(f"Entry Price: ${self.position.avg_price:.2f}")
            print(f"Profit: ${profit:.2f} ({profit_pct:.2f}%)")
            print(f"Cash After Sale: ${self.cash:.2f}")
            print(f"\n📋 MANUAL ORDER DETAILS:")
            print(f"   Order Type: MARKET SELL")
            print(f"   Symbol: {self.symbol}")
            print(f"   Quantity: {self.position.units:.4f}")
            print(f"   Estimated Price: ${price:.2f}")
            print(f"{'='*60}\n")
            
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
    
    def print_status(self, current_price: float):
        """Print current trading status"""
        position_value = self.position.units * current_price if self.position.is_open else 0
        total_equity = self.cash + position_value
        
        print(f"\n{'='*60}")
        print(f"Time: {datetime.now()} | Bar: {self.last_bar_time}")
        print(f"Symbol: {self.symbol} | Price: ${current_price:.2f}")
        print(f"SIMULATED Account - Equity: ${total_equity:.2f} | Cash: ${self.cash:.2f}")
        print(f"Return: {((total_equity - self.initial_capital) / self.initial_capital * 100):.2f}%")
        
        if self.position.is_open:
            unrealized_pnl = position_value - self.position.cost_basis
            unrealized_pct = (unrealized_pnl / self.position.cost_basis) * 100
            print(f"Position: {self.position.units:.4f} @ ${self.position.avg_price:.2f}")
            print(f"Position Value: ${position_value:.2f}")
            print(f"Unrealized P&L: ${unrealized_pnl:.2f} ({unrealized_pct:.2f}%)")
        else:
            print("Position: None")
            
        if self.trades:
            total_profit = sum(t['profit'] for t in self.trades)
            win_rate = sum(1 for t in self.trades if t['profit'] > 0) / len(self.trades) * 100
            print(f"\nTotal Trades: {len(self.trades)} | Win Rate: {win_rate:.1f}%")
            print(f"Total Profit: ${total_profit:.2f}")
            
        if len(self.optimization_history) > 1:
            recent_opt = self.optimization_history[-5:]
            avg_fitness = sum(opt['fitness'] for opt in recent_opt) / len(recent_opt)
            avg_time = sum(opt['elapsed_seconds'] for opt in recent_opt) / len(recent_opt)
            print(f"\nOptimization Stats (last 5):")
            print(f"  Avg Fitness: {avg_fitness:.4f}")
            print(f"  Avg Time: {avg_time:.1f}s")
    
    def run(self):
        """Main trading loop - bar driven"""
        print(f"Starting adaptive live paper trading for {self.symbol}")
        print(f"Timeframe: {self.timeframe_minutes} minute bars")
        print(f"GA Settings: Population={self.ga_population}, Generations={self.ga_generations}")
        print(f"Optimization lookback: {self.ga_lookback_bars} bars")
        print(f"Cache file: {self.csv_filename.name}")
        if not self.cached_df.empty:
            print(f"Loaded {len(self.cached_df)} cached bars from {self.cached_df['timestamp'].min()} to {self.cached_df['timestamp'].max()}")
        print("\nWaiting for first bar...")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        # Track bar timing
        next_bar_time = None
        check_interval = 5  # Check every 5 seconds for new bars
        
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
                df = self.get_historical_data(lookback_bars=max(self.ga_lookback_bars, 250))
                
                # Check if new bar arrived
                if self.check_new_bar(df):
                    print(f"\n{'='*40}")
                    print(f"New {self.timeframe_minutes}-min bar at {self.last_bar_time}")
                    
                    # Run optimization on new bar
                    self.current_params = self.run_optimization(df)
                    
                    # Create/update strategy with mapped parameters
                    # Map GA parameters to strategy parameters
                    strategy_params = {
                        'initial_capital': self.initial_capital,
                        'max_position_fraction': self.current_params.get('max_position_fraction', 1.0),
                        'stop_loss_multiplier_strong': self.current_params.get('stop_loss_pct', 2.0),
                        'stop_loss_multiplier_weak': self.current_params.get('stop_loss_pct', 1.0) * 0.5,
                        'allow_shorts': False  # Paper trading without shorts for now
                    }
                    
                    self.strategy = EnhancedTradingStrategy(**strategy_params)
                    
                    # Check for signals
                    buy_signal, sell_signal, reason = self.check_signals(df)
                    
                    # Get current price
                    current_price = self.get_current_price()
                    
                    # Execute trades
                    if buy_signal and not self.position.is_open:
                        # Calculate position size
                        position_value = self.cash * 0.9  # Use 90% of cash
                        quantity = position_value / current_price
                        
                        if self.is_crypto:
                            quantity = round(quantity, 4)
                        else:
                            quantity = int(quantity)
                        
                        if quantity > 0:
                            self.execute_trade(OrderSide.BUY, quantity, current_price, reason)
                    
                    elif sell_signal and self.position.is_open:
                        self.execute_trade(OrderSide.SELL, self.position.units, current_price, reason)
                    
                    # Update tracking
                    self.update_equity_curve(current_price)
                    self.print_status(current_price)
                    
                    # Show time to next bar
                    time_to_next = (next_bar_time - datetime.now(timezone.utc)).total_seconds()
                    print(f"\nNext bar in {int(time_to_next)}s at {next_bar_time.strftime('%H:%M:%S')}")
                
                # Wait before checking again
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\n\nStopping adaptive trader...")
                # Close any open positions
                if self.position.is_open:
                    print("Closing open position...")
                    current_price = self.get_current_price()
                    self.execute_trade(OrderSide.SELL, self.position.units, current_price, "Manual close")
                break
                
            except Exception as e:
                print(f"\nError in trading loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description='Adaptive Live Paper Trading with GA')
    parser.add_argument('symbol', help='Trading symbol (e.g., MSFT, BTCUSD)')
    parser.add_argument('--api-key', required=True, help='Alpaca API key')
    parser.add_argument('--secret-key', required=True, help='Alpaca secret key')
    parser.add_argument('--crypto', action='store_true', help='Trade cryptocurrency')
    parser.add_argument('--timeframe', type=int, default=1, help='Bar timeframe in minutes (default: 1)')
    
    args = parser.parse_args()
    
    # Create and run trader
    trader = AdaptiveLiveTrader(
        api_key=args.api_key,
        secret_key=args.secret_key,
        symbol=args.symbol,
        is_crypto=args.crypto,
        timeframe_minutes=args.timeframe
    )
    
    try:
        trader.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
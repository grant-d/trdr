#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helios Trader: A quantitative trading algorithm with genetic optimization.

This script loads historical price data, converts it to dollar bars, and can
optionally run a Genetic Algorithm (GA) to find optimal trading parameters
using a more intelligent, constrained parameter space. It then runs a final
backtest simulation and evaluates the performance.

Usage:
  - Backtest with default parameters:
    python helios_trader.py --input_file /path/to/your/data.csv

  - Run GA optimization before the backtest:
    python helios_trader.py --input_file /path/to/your/data.csv --optimize
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import random

# --- Default Configuration ---
DEFAULT_INPUT_FILE = '../data/BTCUSD-feed.csv'
DEFAULT_CAPITAL = 100000.0
DEFAULT_DOLLAR_THRESHOLD = 100_000_000.0

# --- Genetic Algorithm Configuration ---
GA_POPULATION_SIZE = 50 # 20
GA_GENERATIONS = 100 # 10
GA_MUTATION_RATE = 0.2
# GA_WALKFORWARD_PERIODS = 5 # 3
# GA_TRAIN_RATIO = 0.7

# --- Enhanced, Constrained Parameter Space ---
PARAMETER_SPACE = {
    # --- Indicator Selection ---
    'indicator_trend': ['Slope', 'MACD', 'EMA_Crossover'],
    'indicator_volatility': ['ATR', 'StdDev', 'BollingerBandWidth'],
    'indicator_exhaustion': ['SMADiff', 'RSI', 'Stochastic'],

    # --- Dynamic Lookback Periods ---
    'lookback_long': (30, 80, 2),
    'lookback_medium': (15, 40, 1),
    'lookback_short': (5, 20, 1),

    # --- Constrained, Relative Regime Thresholds ---
    'neutral_zone_size': (10.0, 35.0, 1.0),
    'weak_zone_width': (15.0, 50.0, 1.0),

    # --- Risk Management ---
    'stop_loss_multiplier_strong': (1.5, 5.0, 0.25),
    'stop_loss_multiplier_weak': (0.5, 3.0, 0.25),
    'take_profit_multiplier_strong': (2.0, 6.0, 0.25),
    'take_profit_multiplier_weak': (1.0, 4.0, 0.25),
    
    # --- Position Sizing ---
    'entry_step_size': (0.1, 0.5, 0.05),
}

# --- Default Parameters (if not optimizing) ---
DEFAULT_PARAMS = {
    'indicator_trend': 'Slope', 'lookback_long': 50,
    'indicator_volatility': 'ATR', 'lookback_medium': 20,
    'indicator_exhaustion': 'RSI', 'lookback_short': 14,
    'neutral_zone_size': 20.0,
    'weak_zone_width': 30.0,
    'stop_loss_multiplier_strong': 2.5,
    'stop_loss_multiplier_weak': 1.5,
    'take_profit_multiplier_strong': 4.0,
    'take_profit_multiplier_weak': 2.0,
    'entry_step_size': 0.2,
}


class HeliosTrader:
    """ Encapsulates the entire Helios trading algorithm and GA framework. """

    def __init__(self, config):
        """Initializes the trader with a given configuration."""
        self.config = config
        self.raw_df = None
        self.dollar_bars_df = None
        self.results_df = None
        self.best_params = DEFAULT_PARAMS.copy()
        self.indicator_functions = self._get_indicator_functions()
        print("Helios Trader Initialized.")

    # --- Factor Calculation Methods (with new indicators) ---
    def _get_indicator_functions(self):
        return {
            'Trend': {'Slope': self._calculate_trend_slope, 'MACD': self._calculate_trend_macd, 'EMA_Crossover': self._calculate_trend_ema_crossover},
            'Volatility': {'ATR': self._calculate_volatility_atr, 'StdDev': self._calculate_volatility_stddev, 'BollingerBandWidth': self._calculate_volatility_bbw},
            'Exhaustion': {'SMADiff': self._calculate_exhaustion_sma_diff, 'RSI': self._calculate_exhaustion_rsi, 'Stochastic': self._calculate_exhaustion_stochastic}
        }

    def _calculate_trend_slope(self, df, lookback):
        lookback = int(lookback)
        slopes = [linregress(np.arange(lookback), df['Close'].iloc[i-lookback:i]).slope for i in range(lookback, len(df))]
        return pd.Series(slopes, index=df.index[lookback:])

    def _calculate_trend_macd(self, df, lookback, fast=12, slow=26, signal=9):
        # Using lookback_long for slow, medium for fast
        ema_fast = df['Close'].ewm(span=int(self.best_params['lookback_medium']), adjust=False).mean()
        ema_slow = df['Close'].ewm(span=int(self.best_params['lookback_long']), adjust=False).mean()
        return ema_fast - ema_slow

    def _calculate_trend_ema_crossover(self, df, lookback):
        fast_ema = df['Close'].ewm(span=int(self.best_params['lookback_medium']), adjust=False).mean()
        slow_ema = df['Close'].ewm(span=int(self.best_params['lookback_long']), adjust=False).mean()
        return (fast_ema - slow_ema) / df['Close'] # Normalize by price

    def _calculate_volatility_atr(self, df, lookback):
        lookback = int(lookback)
        tr = pd.DataFrame(index=df.index)
        tr['h_l'] = df['High'] - df['Low']
        tr['h_pc'] = abs(df['High'] - df['Close'].shift(1))
        tr['l_pc'] = abs(df['Low'] - df['Close'].shift(1))
        return tr.max(axis=1).rolling(window=lookback).mean()

    def _calculate_volatility_stddev(self, df, lookback):
        return df['Close'].pct_change().rolling(window=int(lookback)).std()

    def _calculate_volatility_bbw(self, df, lookback):
        lookback = int(lookback)
        sma = df['Close'].rolling(window=lookback).mean()
        std = df['Close'].rolling(window=lookback).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        return (upper_band - lower_band) / (sma + 1e-9) # Normalize by SMA

    def _calculate_exhaustion_sma_diff(self, df, lookback, vol_series):
        sma = df['Close'].rolling(window=int(lookback)).mean()
        return (df['Close'] - sma) / (vol_series + 1e-9)

    def _calculate_exhaustion_rsi(self, df, lookback, vol_series):
        lookback = int(lookback)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=lookback - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=lookback - 1, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) * 2 # Normalize to -100 to 100

    def _calculate_exhaustion_stochastic(self, df, lookback, vol_series):
        lookback = int(lookback)
        low_min = df['Low'].rolling(window=lookback).min()
        high_max = df['High'].rolling(window=lookback).max()
        stoch = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
        return (stoch - 50) * 2 # Normalize to -100 to 100

    # --- Core Workflow Methods ---
    def _load_and_prepare_data(self):
        """
        Loads data, handles duplicate columns, and ensures numeric types.
        This is a more robust method to prevent type errors downstream.
        """
        print(f"\n[Step 1/6] Loading data from '{self.config['input_file']}'...")
        try:
            df = pd.read_csv(self.config['input_file'], parse_dates=['Date'], index_col='Date')
            
            # Handle potential duplicate columns by taking the first instance
            df = df.loc[:, ~df.columns.duplicated()]

            # Define columns that must be numeric
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if 'Adj Close' in df.columns:
                numeric_cols.append('Adj Close')

            # Explicitly cast columns to numeric types, coercing errors to NaN
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop any rows where coercion to numeric failed
            df.dropna(subset=numeric_cols, inplace=True)
            
            self.raw_df = df.sort_index()
            self.raw_df = df.sort_index()
            if 'Adj Close' not in self.raw_df.columns:
                self.raw_df['Adj Close'] = self.raw_df['Close']
            print("Data loaded and cleaned successfully.")
        except FileNotFoundError:
            print(f"ERROR: Input file not found at '{self.config['input_file']}'.")
            exit()
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            exit()

    def _create_dollar_bars(self):
        print(f"[Step 2/6] Creating dollar bars with threshold: ${self.config['dollar_threshold']:,}...")
        bars = []
        df = self.raw_df
        threshold = self.config['dollar_threshold']
        
        current_dollar_volume = 0
        open_price, high_price, low_price, close_price = None, -float('inf'), float('inf'), None
        start_time, total_volume = None, 0

        for index, row in df.iterrows():
            if start_time is None:
                start_time = index
                open_price = row['Adj Close']

            dollar_value = row['Adj Close'] * row['Volume']
            current_dollar_volume += dollar_value
            total_volume += row['Volume']
            high_price = max(high_price, row['Adj Close'])
            low_price = min(low_price, row['Adj Close'])
            close_price = row['Adj Close']

            if current_dollar_volume >= threshold:
                bars.append({
                    'Date': index, 'Open': open_price, 'High': high_price,
                    'Low': low_price, 'Close': close_price, 'Volume': total_volume,
                    'Adj Close': close_price
                })
                current_dollar_volume, total_volume = 0, 0
                open_price, high_price, low_price = None, -float('inf'), float('inf')
                start_time = None
                
        self.dollar_bars_df = pd.DataFrame(bars).set_index('Date')
        print(f"Created {len(self.dollar_bars_df)} dollar bars.")
        
    def _calculate_all_factors(self, df, params):
        df_out = df.copy()
        try:
            # Assign lookbacks based on parameter type
            lookback_map = {
                'Trend': params['lookback_long'],
                'Volatility': params['lookback_medium'],
                'Exhaustion': params['lookback_short']
            }
            # Calculate absolute volatility first for stop-loss calculations
            df_out['Abs_Volatility'] = self.indicator_functions['Volatility'][params['indicator_volatility']](df_out, lookback_map['Volatility'])
            
            df_out['Trend'] = self.indicator_functions['Trend'][params['indicator_trend']](df_out, lookback_map['Trend'])
            df_out['Exhaustion'] = self.indicator_functions['Exhaustion'][params['indicator_exhaustion']](df_out, lookback_map['Exhaustion'], df_out['Abs_Volatility'])
            
            # Normalize factors for MSS
            df_out['Norm_Trend'] = (df_out['Trend'].rank(pct=True) - 0.5) * 200
            df_out['Norm_Volatility'] = (df_out['Abs_Volatility'].rank(pct=True) - 0.5) * 200
            df_out['Norm_Exhaustion'] = (df_out['Exhaustion'].rank(pct=True) - 0.5) * 200

        except KeyError as e:
            print(f"ERROR: Invalid indicator name in params: {e}")
            return None
            
        return df_out.dropna()

    def _run_simulation(self, df_with_factors, params):
        """
        Runs the full, sophisticated backtest simulation using the logic
        from the original notebook.
        """
        df = df_with_factors.copy()
        
        # --- Initialize Simulation State ---
        initial_capital = self.config['initial_capital']
        current_capital = initial_capital
        position_size = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        peak_price_since_entry = -float('inf')
        valley_price_since_entry = float('inf')
        equity_curve = []
        trade_log = []

        # --- Get Parameters from GA or Defaults ---
        weights = {'trend': 0.5, 'volatility': 0.2, 'exhaustion': 0.3} # Fixed weights for now
        neutral_size = params.get('neutral_zone_size', 20.0)
        weak_width = params.get('weak_zone_width', 30.0)
        stop_mult_strong = params.get('stop_loss_multiplier_strong', 2.5)
        stop_mult_weak = params.get('stop_loss_multiplier_weak', 1.5)
        tp_mult_strong = params.get('take_profit_multiplier_strong', 4.0)
        tp_mult_weak = params.get('take_profit_multiplier_weak', 2.0)
        entry_step = params.get('entry_step_size', 0.2)
        max_pos_fraction = 1.0

        # --- Calculate MSS and Regimes ---
        df['MSS'] = (weights['trend'] * df['Norm_Trend'] +
                     weights['volatility'] * df['Norm_Volatility'] +
                     weights['exhaustion'] * df['Norm_Exhaustion'])

        strong_bull_thresh = neutral_size + weak_width
        weak_bull_thresh = neutral_size
        weak_bear_thresh = -neutral_size
        strong_bear_thresh = -neutral_size - weak_width

        def classify_regime(mss):
            if mss > strong_bull_thresh: return 'Strong Bull'
            if mss > weak_bull_thresh: return 'Weak Bull'
            if mss < strong_bear_thresh: return 'Strong Bear'
            if mss < weak_bear_thresh: return 'Weak Bear'
            return 'Neutral'
        df['Regime'] = df['MSS'].apply(classify_regime)

        # --- Main Simulation Loop ---
        for index, row in df.iterrows():
            price = row['Close']
            regime = row['Regime']
            mss = row['MSS']
            abs_vol = row['Abs_Volatility']

            # --- Target Position Sizing ---
            target_pos = 0.0
            if regime == 'Strong Bull':
                target_pos = 1.0
            elif regime == 'Strong Bear':
                target_pos = -1.0
            elif regime == 'Neutral':
                target_pos = 0.0
            
            # --- Gradual Entry/Exit ---
            pos_change = np.clip(target_pos - position_size, -entry_step, entry_step)
            
            if abs(pos_change) > 1e-9:
                if position_size != 0 and np.sign(pos_change) != np.sign(position_size): # Reducing position
                    pnl = (price - entry_price) * pos_change
                    current_capital += pnl
                    trade_log.append(f"{index}: Reduce Position @ {price:.2f}, PnL: {pnl:.2f}")
                else: # Increasing position
                    new_total_size = position_size + pos_change
                    entry_price = (entry_price * abs(position_size) + price * abs(pos_change)) / abs(new_total_size)
                    trade_log.append(f"{index}: Change Position @ {price:.2f}")
                position_size += pos_change

            # --- Risk Management ---
            if position_size > 0: # Long
                stop_dist = abs_vol * (stop_mult_strong if regime == 'Strong Bull' else stop_mult_weak)
                tp_dist = abs_vol * (tp_mult_strong if regime == 'Strong Bull' else tp_mult_weak)
                peak_price_since_entry = max(peak_price_since_entry, price)
                stop_loss = max(stop_loss, peak_price_since_entry - stop_dist)
                take_profit = entry_price + tp_dist
                
                if price <= stop_loss or price >= take_profit:
                    pnl = (price - entry_price) * position_size
                    current_capital += pnl
                    trade_log.append(f"{index}: Exit Long @ {price:.2f}, PnL: {pnl:.2f}")
                    position_size, entry_price, stop_loss, take_profit = 0, 0, 0, 0
                    peak_price_since_entry = -float('inf')

            elif position_size < 0: # Short
                stop_dist = abs_vol * (stop_mult_strong if regime == 'Strong Bear' else stop_mult_weak)
                tp_dist = abs_vol * (tp_mult_strong if regime == 'Strong Bear' else tp_mult_weak)
                valley_price_since_entry = min(valley_price_since_entry, price)
                stop_loss = min(stop_loss, valley_price_since_entry + stop_dist)
                take_profit = entry_price - tp_dist
                
                if price >= stop_loss or price <= take_profit:
                    pnl = (entry_price - price) * abs(position_size)
                    current_capital += pnl
                    trade_log.append(f"{index}: Exit Short @ {price:.2f}, PnL: {pnl:.2f}")
                    position_size, entry_price, stop_loss, take_profit = 0, 0, 0, 0
                    valley_price_since_entry = float('inf')

            # --- Update Equity Curve ---
            unrealized_pnl = (price - entry_price) * position_size if entry_price > 0 else 0
            equity_curve.append(current_capital + unrealized_pnl)

        df['Equity'] = equity_curve
        return df

    def _fitness_function(self, params, df_train):
        df_with_factors = self._calculate_all_factors(df_train, params)
        if df_with_factors is None or df_with_factors.empty:
            return -1000

        sim_results = self._run_simulation(df_with_factors, params)
        if sim_results.empty:
            return -1000
            
        returns = sim_results['Equity'].pct_change()
        downside_std = returns[returns < 0].std()
        
        sortino = (returns.mean() * (252**0.5)) / (downside_std + 1e-9)
        return sortino if np.isfinite(sortino) else -1000

    def _generate_random_param(self, param, space):
        if isinstance(space, list):
            return random.choice(space)
        elif isinstance(space, tuple) and len(space) == 3:
            min_val, max_val, step = space
            if isinstance(min_val, int):
                return random.randrange(min_val, max_val + step, step)
            elif isinstance(min_val, float):
                val = random.uniform(min_val, max_val)
                return round(round(val / step) * step, 8)
        return None

    def _run_ga_optimization(self):
        print("\n[Step 3/6] Running Genetic Algorithm Optimization...")
        
        population = [{p: self._generate_random_param(p, s) for p, s in PARAMETER_SPACE.items()} for _ in range(GA_POPULATION_SIZE)]
        best_fitness = -float('inf')

        for gen in range(GA_GENERATIONS):
            fitnesses = [self._fitness_function(p, self.dollar_bars_df) for p in population]
            
            best_gen_idx = np.argmax(fitnesses)
            if fitnesses[best_gen_idx] > best_fitness:
                best_fitness = fitnesses[best_gen_idx]
                self.best_params = population[best_gen_idx].copy()
            
            new_population = [self.best_params]
            for _ in range(1, GA_POPULATION_SIZE):
                parent = random.choice(population)
                mutated_child = parent.copy()
                param_to_mutate = random.choice(list(PARAMETER_SPACE.keys()))
                mutated_child[param_to_mutate] = self._generate_random_param(param_to_mutate, PARAMETER_SPACE[param_to_mutate])
                new_population.append(mutated_child)
            population = new_population
            
            print(f"  Generation {gen+1}/{GA_GENERATIONS}, Best Fitness (Sortino): {best_fitness:.4f}")

        print("GA optimization complete.")
        print(f"Best parameters found: {self.best_params}")

    def _run_final_backtest(self):
        print(f"\n[Step 4/6] Running final backtest with best parameters...")
        self.results_df = self._calculate_all_factors(self.dollar_bars_df, self.best_params)
        self.results_df = self._run_simulation(self.results_df, self.best_params)
        print("Final backtest finished.")

    def _evaluate_performance(self):
        print("\n[Step 5/6] Performance Evaluation:")
        if self.results_df is None or self.results_df.empty:
            print("  No results to evaluate.")
            return
        
        initial_capital = self.config['initial_capital']
        final_equity = self.results_df['Equity'].iloc[-1]
        
        total_return = (final_equity - initial_capital) / initial_capital
        returns = self.results_df['Equity'].pct_change()
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (returns.mean() * (252**0.5)) / (downside_std + 1e-9)
        
        peak = self.results_df['Equity'].cummax()
        drawdown = (self.results_df['Equity'] - peak) / peak
        max_drawdown = drawdown.min()

        print(f"  - Final Equity: ${final_equity:,.2f}")
        print(f"  - Total Return: {total_return:.2%}")
        print(f"  - Max Drawdown: {max_drawdown:.2%}")
        print(f"  - Sortino Ratio: {sortino_ratio:.2f}")

    def _plot_equity_curve(self):
        print("\n[Step 6/6] Plotting equity curve...")
        plt.figure(figsize=(14, 7))
        plt.plot(self.results_df.index, self.results_df['Equity'], label='Helios Trader Equity')
        plt.title('Helios Trader Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self):
        self._load_and_prepare_data()
        self._create_dollar_bars()
        if self.config.get('optimize'):
            self._run_ga_optimization()
        self._run_final_backtest()
        self._evaluate_performance()
        if self.config.get('plot'):
            self._plot_equity_curve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Helios Trader - A quantitative trading algorithm.")
    parser.add_argument('--input_file', type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument('--initial_capital', type=float, default=DEFAULT_CAPITAL)
    parser.add_argument('--dollar_threshold', type=float, default=DEFAULT_DOLLAR_THRESHOLD)
    parser.add_argument('--optimize', action='store_true', help="Run Genetic Algorithm to find optimal parameters.")
    parser.add_argument('--plot', action='store_true', help="Display the equity curve plot.")
    args = parser.parse_args()

    config = {
        'input_file': args.input_file,
        'initial_capital': args.initial_capital,
        'dollar_threshold': args.dollar_threshold,
        'optimize': args.optimize,
        'plot': args.plot,
    }

    trader = HeliosTrader(config)
    trader.run()

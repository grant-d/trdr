#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helios Trader: A quantitative trading algorithm with genetic optimization.

This script loads historical price data, converts it to dollar bars, and can
optionally run a Genetic Algorithm (GA) with walk-forward optimization to find
optimal trading parameters using a more intelligent, constrained parameter space.
It then runs a final backtest simulation and evaluates the performance.

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
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 100
GA_MUTATION_RATE = 0.2
GA_WALKFORWARD_PERIODS = 3 # Reduced to create larger, more robust windows
GA_TRAIN_RATIO = 0.7

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
            df = df.dropna(subset=numeric_cols)
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
        peak_price, valley_price = -float('inf'), float('inf')
        equity_curve, trade_log = [], []

        # --- Get Parameters from GA or Defaults ---
        weights = {'trend': 0.5, 'volatility': 0.2, 'exhaustion': 0.3}
        neutral_size = params.get('neutral_zone_size', 20.0)
        weak_width = params.get('weak_zone_width', 30.0)
        stop_mult_strong = params.get('stop_loss_multiplier_strong', 2.5)
        stop_mult_weak = params.get('stop_loss_multiplier_weak', 1.5)
        tp_mult_strong = params.get('take_profit_multiplier_strong', 4.0)
        tp_mult_weak = params.get('take_profit_multiplier_weak', 2.0)
        entry_step = params.get('entry_step_size', 0.2)

         # --- Calculate MSS and Regimes ---
        df['MSS'] = (weights['trend'] * df['Norm_Trend'] +
                     weights['volatility'] * df['Norm_Volatility'] +
                     weights['exhaustion'] * df['Norm_Exhaustion'])

        strong_bull_thresh, weak_bull_thresh = neutral_size + weak_width, neutral_size
        weak_bear_thresh, strong_bear_thresh = -neutral_size, -neutral_size - weak_width

        def classify_regime(mss):
            if mss > strong_bull_thresh: return 'Strong Bull'
            if mss > weak_bull_thresh: return 'Weak Bull'
            if mss < strong_bear_thresh: return 'Strong Bear'
            if mss < weak_bear_thresh: return 'Weak Bear'
            return 'Neutral'
        df['Regime'] = df['MSS'].apply(classify_regime)

        # --- Main Simulation Loop ---
        for index, row in df.iterrows():
            price, regime, abs_vol = row['Close'], row['Regime'], row['Abs_Volatility']
            target_pos = 0.0
            if regime == 'Strong Bull': target_pos = 1.0
            elif regime == 'Strong Bear': target_pos = -1.0
            
            # --- Gradual Entry/Exit ---
            pos_change = np.clip(target_pos - position_size, -entry_step, entry_step)
            
            if abs(pos_change) > 1e-9:
                units_to_trade = (pos_change * initial_capital) / price
                if abs(position_size) < 1e-9: # New position
                    entry_price = price
                else: # Adjusting position
                    if np.sign(pos_change) != np.sign(position_size): # Reducing
                        pnl = (price - entry_price) * units_to_trade
                        current_capital += pnl
                    else: # Increasing
                        new_total_size = position_size + pos_change
                        if abs(new_total_size) > 1e-9:
                            entry_price = (entry_price * abs(position_size) + price * abs(pos_change)) / abs(new_total_size)
                position_size += pos_change
                trade_log.append(index)

             # --- Risk Management ---
            if position_size > 0:
                stop_dist = abs_vol * (stop_mult_strong if regime == 'Strong Bull' else stop_mult_weak)
                tp_dist = abs_vol * (tp_mult_strong if regime == 'Strong Bull' else tp_mult_weak)
                peak_price = max(peak_price, price)
                stop_loss = max(stop_loss, peak_price - stop_dist) if stop_loss != 0 else peak_price - stop_dist
                take_profit = entry_price + tp_dist
                if (stop_loss > 0 and price <= stop_loss) or (take_profit > 0 and price >= take_profit):
                    pnl = (price - entry_price) * position_size * initial_capital / entry_price
                    current_capital += pnl
                    position_size, entry_price, stop_loss, take_profit, peak_price = 0, 0, 0, 0, -float('inf')
            elif position_size < 0: # Short
                stop_dist = abs_vol * (stop_mult_strong if regime == 'Strong Bear' else stop_mult_weak)
                tp_dist = abs_vol * (tp_mult_strong if regime == 'Strong Bear' else tp_mult_weak)
                valley_price = min(valley_price, price)
                stop_loss = min(stop_loss, valley_price + stop_dist) if stop_loss != 0 else valley_price + stop_dist
                take_profit = entry_price - tp_dist
                if (stop_loss > 0 and price >= stop_loss) or (take_profit > 0 and price <= take_profit):
                    pnl = (entry_price - price) * abs(position_size) * initial_capital / entry_price
                    current_capital += pnl
                    position_size, entry_price, stop_loss, take_profit, valley_price = 0, 0, 0, 0, float('inf')

            # --- Update Equity Curve ---
            unrealized_pnl = (price - entry_price) * position_size * initial_capital / entry_price if entry_price > 0 else 0
            equity_curve.append(current_capital + unrealized_pnl)

        df['Equity'] = equity_curve
        df['Trade_Count'] = len(trade_log)
        return df

    def _fitness_function(self, params, df_train):
        # A more robust fitness function to prevent overfitting
        min_trades_required = 5
        max_sortino_cap = 10.0

        df_with_factors = self._calculate_all_factors(df_train, params)
        if df_with_factors is None or df_with_factors.empty: return -1000

        sim_results = self._run_simulation(df_with_factors, params)
        if sim_results.empty or 'Trade_Count' not in sim_results.columns: return -1000
            
        if sim_results['Trade_Count'].iloc[0] < min_trades_required:
            return -500 # Penalize for not trading enough

        returns = sim_results['Equity'].pct_change()
        downside_std = returns[returns < 0].std()
        
        sortino = (returns.mean() * (252**0.5)) / (downside_std + 1e-9)
        
        # Cap the sortino to prevent chasing extreme, overfitted values
        capped_sortino = min(sortino, max_sortino_cap)
        
        return capped_sortino if np.isfinite(capped_sortino) else -1000

    def _generate_random_param(self, param, space):
        if isinstance(space, list): return random.choice(space)
        elif isinstance(space, tuple) and len(space) == 3:
            min_val, max_val, step = space
            if isinstance(min_val, int): return random.randrange(min_val, max_val + step, step)
            elif isinstance(min_val, float):
                val = random.uniform(min_val, max_val)
                return round(round(val / step) * step, 8)
        return None

    def _run_ga_optimization(self):
        print("\n[Step 3/6] Running Genetic Algorithm with Walk-Forward Optimization...")
        
        all_data = self.dollar_bars_df
        n_periods = self.config.get('walkforward_periods', GA_WALKFORWARD_PERIODS)
        train_ratio = self.config.get('train_ratio', GA_TRAIN_RATIO)
        
        window_size = len(all_data) // n_periods
        train_size = int(window_size * train_ratio)
        
        overall_best_test_fitness = -float('inf')
        
        for i in range(n_periods):
            start_idx = i * window_size
            train_end_idx = start_idx + train_size
            test_end_idx = train_end_idx + (window_size - train_size)
            
            if test_end_idx > len(all_data): continue

            train_data = all_data.iloc[start_idx:train_end_idx]
            test_data = all_data.iloc[train_end_idx:test_end_idx]
            
            print(f"\n--- Walk-Forward Window {i+1}/{n_periods} ---")
            print(f"  Training on {len(train_data)} bars...")

            population = [{p: self._generate_random_param(p, s) for p, s in PARAMETER_SPACE.items()} for _ in range(GA_POPULATION_SIZE)]
            best_window_fitness = -float('inf')
            best_window_params = None

            for gen in range(GA_GENERATIONS):
                fitnesses = [self._fitness_function(p, train_data) for p in population]
                best_gen_idx = np.argmax(fitnesses)
                if fitnesses[best_gen_idx] > best_window_fitness:
                    best_window_fitness = fitnesses[best_gen_idx]
                    best_window_params = population[best_gen_idx].copy()
                
                new_population = [best_window_params]
                for _ in range(1, GA_POPULATION_SIZE):
                    parent = random.choice(population)
                    mutated_child = parent.copy()
                    param_to_mutate = random.choice(list(PARAMETER_SPACE.keys()))
                    mutated_child[param_to_mutate] = self._generate_random_param(param_to_mutate, PARAMETER_SPACE[param_to_mutate])
                    new_population.append(mutated_child)
                population = new_population
            
            print(f"  Best Train Fitness (Sortino) for window {i+1}: {best_window_fitness:.4f}")
            test_fitness = self._fitness_function(best_window_params, test_data)
            print(f"  Test Fitness on unseen data: {test_fitness:.4f}")
            
            if test_fitness > overall_best_test_fitness:
                overall_best_test_fitness = test_fitness
                self.best_params = best_window_params.copy()
                print(f"  *** New Overall Best Parameters Found! ***")

        print("\nGA optimization complete.")
        print(f"Best parameters found across all walk-forward tests: {self.best_params}")

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
        downside_std = returns[returns < 0].std()
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

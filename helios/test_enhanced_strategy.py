"""
Test enhanced strategy vs basic strategy
"""

import pandas as pd
import numpy as np
from data_processing import create_dollar_bars, prepare_data
from factors import calculate_mss, calculate_macd, calculate_rsi
from strategy import TradingStrategy
from strategy_enhanced import EnhancedTradingStrategy
from performance import generate_performance_report

# Load test data
print("Loading data...")
df = pd.read_csv('./data/BTCUSD.csv', parse_dates=True, index_col=0)
df.columns = df.columns.str.lower()
df = prepare_data(df)

# Create dollar bars
print("Creating dollar bars...")
df = create_dollar_bars(df, 1000000)

# Calculate factors
print("Calculating factors...")
factors_df, regimes = calculate_mss(df, lookback=20)

# Add MACD and RSI
macd_data = calculate_macd(df)
factors_df['macd_hist'] = macd_data['histogram']
factors_df['rsi'] = calculate_rsi(df)

# Merge
combined_df = pd.concat([df, factors_df], axis=1)

# Test basic strategy
print("\n" + "="*60)
print("Testing BASIC Strategy")
print("="*60)

basic_strategy = TradingStrategy(
    initial_capital=100000,
    max_position_pct=0.95,
    min_position_pct=0.1
)

basic_results = basic_strategy.run_backtest(combined_df, combined_df)
basic_trades = basic_strategy.get_trade_summary()
basic_report = generate_performance_report(basic_results, basic_trades, 100000)
print(basic_report)

# Test enhanced strategy with default parameters
print("\n" + "="*60)
print("Testing ENHANCED Strategy (Default Parameters)")
print("="*60)

enhanced_strategy = EnhancedTradingStrategy(
    initial_capital=100000,
    max_position_fraction=1.0,
    entry_step_size=0.2,
    stop_loss_multiplier_strong=2.0,
    stop_loss_multiplier_weak=1.0,
    strong_bull_threshold=50.0,
    weak_bull_threshold=20.0,
    neutral_upper=20.0,
    neutral_lower=-20.0,
    weak_bear_threshold=-20.0,
    strong_bear_threshold=-50.0,
)

enhanced_results = enhanced_strategy.run_backtest(combined_df, combined_df)
enhanced_trades = enhanced_strategy.get_trade_summary()
enhanced_report = generate_performance_report(enhanced_results, enhanced_trades, 100000)
print(enhanced_report)

# Test enhanced strategy with optimized parameters from old code
print("\n" + "="*60)
print("Testing ENHANCED Strategy (Old Code Optimized Parameters)")
print("="*60)

# These are example optimized parameters from the old code
optimized_strategy = EnhancedTradingStrategy(
    initial_capital=100000,
    max_position_fraction=1.0,
    entry_step_size=0.3,  # Gradual entry
    stop_loss_multiplier_strong=3.0,  # Wider stops as per old code
    stop_loss_multiplier_weak=2.0,
    strong_bull_threshold=40.0,  # Adjusted thresholds
    weak_bull_threshold=15.0,
    neutral_upper=15.0,
    neutral_lower=-15.0,
    weak_bear_threshold=-15.0,
    strong_bear_threshold=-40.0,
)

optimized_results = optimized_strategy.run_backtest(combined_df, combined_df)
optimized_trades = optimized_strategy.get_trade_summary()
optimized_report = generate_performance_report(optimized_results, optimized_trades, 100000)
print(optimized_report)

# Debug enhanced strategy
print("\n" + "="*60)
print("DEBUG: Enhanced Strategy Analysis")
print("="*60)

# Check position behavior
print("\nPosition Analysis:")
print(f"Max position fraction in results: {enhanced_results['position_fraction'].max():.4f}")
print(f"Min position fraction in results: {enhanced_results['position_fraction'].min():.4f}")
print(f"Average position fraction: {enhanced_results['position_fraction'].abs().mean():.4f}")

# Check regime distribution
print("\nRegime Distribution:")
print(enhanced_results['regime'].value_counts())

# Check if trades are happening
print(f"\nTotal trades executed: {len(enhanced_strategy.trades)}")
print(f"Trades with P&L: {len([t for t in enhanced_strategy.trades if t.pnl != 0])}")

# Compare key metrics
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

def extract_metric(report, metric_name):
    for line in report.split('\n'):
        if metric_name in line:
            try:
                return float(line.split()[-1].replace('%', ''))
            except:
                return 0.0
    return 0.0

metrics = ['Total Return:', 'Sortino Ratio:', 'Maximum Drawdown:', 'Win Rate:']
strategies = [
    ('Basic', basic_report),
    ('Enhanced Default', enhanced_report),
    ('Enhanced Optimized', optimized_report)
]

for metric in metrics:
    print(f"\n{metric}")
    for name, report in strategies:
        value = extract_metric(report, metric)
        print(f"  {name:20s}: {value:8.2f}{'%' if 'Return' in metric or 'Drawdown' in metric or 'Rate' in metric else ''}")
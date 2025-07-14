"""
Check MSS distribution to understand position sizing issues
"""

import pandas as pd
import numpy as np
from data_processing import create_dollar_bars, prepare_data
from factors import calculate_mss

# Load test data
df = pd.read_csv('/Users/grantdickinson/repos/trdr/data/BTCUSD-feed.csv', parse_dates=True, index_col=0)
df.columns = df.columns.str.lower()
df = prepare_data(df)

# Create dollar bars
df = create_dollar_bars(df, 1000000)

# Calculate factors
factors_df, regimes = calculate_mss(df, lookback=20)

print("MSS Statistics:")
print(f"Mean: {factors_df['mss'].mean():.2f}")
print(f"Std: {factors_df['mss'].std():.2f}")
print(f"Min: {factors_df['mss'].min():.2f}")
print(f"Max: {factors_df['mss'].max():.2f}")

print("\nMSS Percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"{p}%: {factors_df['mss'].quantile(p/100):.2f}")

print("\nRegime Distribution:")
print(regimes.value_counts())
print(f"\nTotal bars: {len(regimes)}")

# Check MSS distribution within Weak Bull regime
weak_bull_mss = factors_df[regimes == 'Weak Bull']['mss']
print("\nWeak Bull MSS Distribution:")
print(f"Mean: {weak_bull_mss.mean():.2f}")
print(f"Min: {weak_bull_mss.min():.2f}")
print(f"Max: {weak_bull_mss.max():.2f}")

# Simulate position sizing for Weak Bull
print("\nSimulated Position Sizing for Weak Bull:")
weak_bull_threshold = 20.0
neutral_upper = 20.0

# Fix the calculation - weak bull goes from 20 to 50 (next threshold)
weak_bull_upper = 50.0  # The upper bound of weak bull is the strong bull threshold

for mss in [20, 25, 30, 35, 40, 45, 50]:
    if mss >= weak_bull_threshold and mss <= weak_bull_upper:
        normalized = (mss - weak_bull_threshold) / (weak_bull_upper - weak_bull_threshold)
        position_fraction = 0.5 * normalized  # 0.5 factor for weak regime
        print(f"MSS={mss}: normalized={normalized:.3f}, position_fraction={position_fraction:.3f}")
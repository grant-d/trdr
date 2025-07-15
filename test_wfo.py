#!/usr/bin/env python3
"""
Test script for Walk-Forward Optimizer.

Demonstrates how to use the WalkForwardOptimizer class with market data.
"""

import pandas as pd
import sys
from walk_forward_optimizer import WalkForwardOptimizer


def main():
    """Test the Walk-Forward Optimizer with sample data."""
    
    # Try to load actual data if available
    try:
        df = pd.read_csv('data/btc_usd_1m_bars.csv', parse_dates=['timestamp'])
        print(f"Loaded {len(df)} bars from btc_usd_1m_bars.csv")
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        print("Creating sample data for demonstration...")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + pd.Series(range(len(dates))) * 0.1,
            'high': 101 + pd.Series(range(len(dates))) * 0.1,
            'low': 99 + pd.Series(range(len(dates))) * 0.1,
            'close': 100.5 + pd.Series(range(len(dates))) * 0.1,
            'volume': 1000 + pd.Series(range(len(dates))) * 10
        })
    
    print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total samples: {len(df)}")
    
    # Test 1: Default settings (5 splits, expanding window)
    print("\n" + "="*80)
    print("Test 1: Default WFO (5 splits, expanding window)")
    print("="*80)
    
    wfo = WalkForwardOptimizer(n_splits=5)
    
    if not wfo.validate_data_sufficiency(df):
        print("Insufficient data for requested splits")
        return
        
    splits = wfo.get_splits(df)
    wfo.print_splits_summary(splits)
    
    # Calculate and display statistics
    stats = wfo.calculate_split_statistics(splits)
    print("\nSplit Statistics:")
    print(f"  Average training size: {stats['avg_train_size']:.0f} bars")
    print(f"  Average test size: {stats['avg_test_size']:.0f} bars")
    print(f"  Total training bars: {stats['total_train_bars']:,}")
    print(f"  Total test bars: {stats['total_test_bars']:,}")
    print(f"  Train/Test ratio: {stats['train_test_ratio']:.2f}")
    
    # Test 2: Fixed test size
    print("\n" + "="*80)
    print("Test 2: Fixed test size (100 samples per test)")
    print("="*80)
    
    wfo_fixed = WalkForwardOptimizer(n_splits=5, test_size=100)
    
    if not wfo_fixed.validate_data_sufficiency(df):
        print("Insufficient data for requested splits")
        return
        
    splits_fixed = wfo_fixed.get_splits(df)
    wfo_fixed.print_splits_summary(splits_fixed)
    
    # Test 3: With gap between train and test
    print("\n" + "="*80)
    print("Test 3: With gap (10 samples) between train and test")
    print("="*80)
    
    wfo_gap = WalkForwardOptimizer(n_splits=3, test_size=50, gap=10)
    
    if not wfo_gap.validate_data_sufficiency(df):
        print("Insufficient data for requested splits")
        return
        
    splits_gap = wfo_gap.get_splits(df)
    wfo_gap.print_splits_summary(splits_gap)
    
    # Demonstrate getting actual data for a split
    print("\n" + "="*80)
    print("Example: Extracting data for Split 1")
    print("="*80)
    
    train_df, test_df = wfo.get_split_data(df, splits[0])
    
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Training indices: {splits[0].train_start_idx} to {splits[0].train_end_idx}")
    if 'timestamp' in train_df.columns:
        print(f"Training time range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    
    print(f"\nTest data shape: {test_df.shape}")
    print(f"Test indices: {splits[0].test_start_idx} to {splits[0].test_end_idx}")
    if 'timestamp' in test_df.columns:
        print(f"Test time range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # Show first few rows of each
    print("\nFirst 3 rows of training data:")
    print(train_df.head(3))
    
    print("\nFirst 3 rows of test data:")
    print(test_df.head(3))


if __name__ == "__main__":
    main()
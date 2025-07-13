#!/usr/bin/env python3
"""
Example usage of Helios Trading Analysis Toolkit
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import Helios modules
from data_processing import create_dollar_bars, prepare_data
from factors import calculate_macd, calculate_rsi, calculate_mss
from strategy import TradingStrategy
from performance import evaluate_strategy_performance, generate_performance_report


def generate_sample_data(days=252*2):
    """Generate sample OHLCV data for testing"""
    print("Generating sample OHLCV data...")
    
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate synthetic price data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some intraday volatility
        daily_vol = abs(np.random.normal(0, 0.01))
        high = price * (1 + daily_vol)
        low = price * (1 - daily_vol)
        open_price = price * (1 + np.random.normal(0, daily_vol/2))
        volume = np.random.lognormal(15, 0.5)  # Log-normal volume
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def run_example():
    """Run example Helios analysis"""
    
    # 1. Generate or load data
    df = generate_sample_data()
    print(f"Generated {len(df)} days of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # 2. Prepare data
    print("\nPreparing data...")
    df = prepare_data(df)
    
    # 3. Create dollar bars (optional)
    print("\nCreating dollar bars...")
    dollar_bars = create_dollar_bars(df, dollar_threshold=500000)
    print(f"Created {len(dollar_bars)} dollar bars from {len(df)} daily bars")
    
    # Use dollar bars for analysis
    analysis_df = dollar_bars.copy()
    
    # 4. Calculate indicators
    print("\nCalculating technical indicators...")
    
    # MACD
    macd_data = calculate_macd(analysis_df)
    analysis_df['macd'] = macd_data['macd']
    analysis_df['macd_signal'] = macd_data['signal']
    analysis_df['macd_hist'] = macd_data['histogram']
    
    # RSI
    analysis_df['rsi'] = calculate_rsi(analysis_df)
    
    # 5. Calculate Market State Score
    print("\nCalculating Market State Score (MSS)...")
    factors_df, regimes = calculate_mss(analysis_df, lookback=20)
    
    # Merge with main dataframe
    analysis_df = pd.concat([analysis_df, factors_df], axis=1)
    
    print("\nRegime distribution:")
    print(regimes.value_counts().sort_index())
    
    # 6. Run trading strategy
    print("\nRunning trading strategy backtest...")
    initial_capital = 100000
    strategy = TradingStrategy(
        initial_capital=initial_capital,
        max_position_pct=0.95,
        min_position_pct=0.1
    )
    
    results = strategy.run_backtest(analysis_df, analysis_df)
    
    # 7. Evaluate performance
    print("\nEvaluating performance...")
    trades_summary = strategy.get_trade_summary()
    performance_metrics = evaluate_strategy_performance(results, initial_capital)
    
    # Generate report
    report = generate_performance_report(results, trades_summary, initial_capital)
    print("\n" + report)
    
    # 8. Save results
    output_dir = Path('./example_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save data
    analysis_df.to_csv(output_dir / 'analysis_data.csv')
    results.to_csv(output_dir / 'backtest_results.csv')
    
    # Save report
    with open(output_dir / 'performance_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to: {output_dir}")
    
    # 9. Plot results (optional - requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Price and Portfolio Value
        ax1 = axes[0]
        ax1.plot(results.index, results['close'], label='Price', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(results.index, results['portfolio_value'], 
                     label='Portfolio Value', color='green', alpha=0.7)
        ax1.set_ylabel('Price')
        ax1_twin.set_ylabel('Portfolio Value')
        ax1.set_title('Price and Portfolio Value')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: MSS and Regime
        ax2 = axes[1]
        ax2.plot(results.index, results['mss'], label='MSS', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Market State Score')
        ax2.set_title('Market State Score')
        ax2.legend()
        
        # Color background by regime
        regime_colors = {
            'Strong Bull': 'darkgreen',
            'Weak Bull': 'lightgreen',
            'Neutral': 'yellow',
            'Weak Bear': 'orange',
            'Strong Bear': 'red'
        }
        
        for regime, color in regime_colors.items():
            mask = results['regime'] == regime
            ax2.fill_between(results.index, -1, 1, where=mask, 
                           alpha=0.2, color=color, label=regime)
        
        # Plot 3: Position and Trades
        ax3 = axes[2]
        ax3.plot(results.index, results['position'], label='Position', alpha=0.7)
        ax3.set_ylabel('Position (shares)')
        ax3.set_title('Trading Position')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'analysis_plots.png', dpi=150)
        print(f"Plots saved to: {output_dir / 'analysis_plots.png'}")
        
    except ImportError:
        print("\nMatplotlib not installed. Skipping plots.")


if __name__ == "__main__":
    run_example()
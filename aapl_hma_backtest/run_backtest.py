"""
Main Backtest Script

Run the HMA Bull Put Spread strategy backtest on AAPL
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data.data_loader import load_and_prepare_data
from indicators.hma import calculate_hma_signals
from strategy.hma_bull_put_strategy import HMABullPutStrategy


def run_backtest(ticker: str = 'AAPL',
                start_date: str = '2020-01-01',
                end_date: str = '2024-10-31',
                initial_capital: float = 100000,
                save_results: bool = True):
    """
    Run complete backtest

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        save_results: Whether to save results to CSV

    Returns:
        Portfolio object with results
    """
    print("="*80)
    print(f"HMA BULL PUT SPREAD STRATEGY BACKTEST - {ticker}")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("-"*80)

    # Load data
    print("\n[1/5] Loading historical data...")
    df = load_and_prepare_data(ticker, start_date, end_date)
    print(f"   Loaded {len(df)} trading days")
    print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    # Calculate HMA indicators
    print("\n[2/5] Calculating Hull Moving Averages...")
    df = calculate_hma_signals(df, 50)
    df = calculate_hma_signals(df, 200)

    # Remove NaN values from HMA calculation
    df = df.dropna()
    print(f"   Data points after HMA calculation: {len(df)}")

    # Count signals
    entry_signals = (df['HMA_50_UP'] & df['HMA_200_UP']).sum()
    print(f"   Entry signals (both HMAs up): {entry_signals}")

    # Initialize strategy
    print("\n[3/5] Initializing strategy...")
    strategy = HMABullPutStrategy(
        initial_capital=initial_capital,
        short_put_delta=-0.30,  # 30 delta short put
        spread_width=5.0,       # $5 wide spread
        days_to_expiration=30,  # 30 DTE
        profit_target=0.50,     # Take profit at 50%
        stop_loss=2.0,          # Stop at 2x credit
        max_positions=5         # Max 5 concurrent positions
    )
    print("   Strategy parameters:")
    print(f"   - Short Put Delta: {strategy.short_put_delta}")
    print(f"   - Spread Width: ${strategy.spread_width}")
    print(f"   - Days to Expiration: {strategy.days_to_expiration}")
    print(f"   - Profit Target: {strategy.profit_target*100}% of max profit")
    print(f"   - Stop Loss: {strategy.stop_loss}x credit received")
    print(f"   - Max Positions: {strategy.max_positions}")

    # Run backtest
    print("\n[4/5] Running backtest...")
    portfolio = strategy.run_backtest(df)
    print(f"   Backtest complete!")

    # Generate statistics
    print("\n[5/5] Generating results...")
    stats = portfolio.get_statistics()

    # Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)

    if stats:
        print(f"\nTrade Statistics:")
        print(f"  Total Trades:        {stats['total_trades']}")
        print(f"  Winning Trades:      {stats['winning_trades']}")
        print(f"  Losing Trades:       {stats['losing_trades']}")
        print(f"  Win Rate:            {stats['win_rate']*100:.2f}%")
        print(f"  Average Days Held:   {stats['avg_days_held']:.1f}")

        print(f"\nProfit & Loss:")
        print(f"  Total P&L:           ${stats['total_pnl']:,.2f}")
        print(f"  Average Win:         ${stats['avg_win']:,.2f}")
        print(f"  Average Loss:        ${stats['avg_loss']:,.2f}")

        print(f"\nPortfolio Performance:")
        print(f"  Initial Capital:     ${initial_capital:,.2f}")
        print(f"  Final Equity:        ${stats['final_equity']:,.2f}")
        print(f"  Total Return:        {stats['return_pct']:.2f}%")

        # Calculate additional metrics
        if portfolio.equity_curve:
            equity_df = pd.DataFrame(portfolio.equity_curve)
            equity_values = equity_df['total_equity'].values

            # Drawdown
            peak = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - peak) / peak
            max_drawdown = drawdown.min()

            # Sharpe ratio (simplified)
            returns = np.diff(equity_values) / equity_values[:-1]
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe = 0

            print(f"  Max Drawdown:        {max_drawdown*100:.2f}%")
            print(f"  Sharpe Ratio:        {sharpe:.2f}")

        # Save results
        if save_results and portfolio.closed_trades:
            # Use absolute path or script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            reports_dir = os.path.join(script_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)

            trades_df = pd.DataFrame(portfolio.closed_trades)
            trades_file = os.path.join(reports_dir, f'trades_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"\n  Trades saved to: {trades_file}")

            equity_file = os.path.join(reports_dir, f'equity_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            equity_df.to_csv(equity_file, index=False)
            print(f"  Equity curve saved to: {equity_file}")

    else:
        print("\nNo trades were executed during the backtest period.")
        print("This could be due to:")
        print("  - No entry signals generated (HMA 50 and 200 not both up)")
        print("  - Insufficient data for HMA calculation")
        print("  - Date range too short")

    print("\n" + "="*80)

    return portfolio, df


def plot_results(portfolio, df):
    """
    Create visualization plots

    Args:
        portfolio: Portfolio object
        df: DataFrame with price and indicator data
    """
    if not portfolio.closed_trades or not portfolio.equity_curve:
        print("No data to plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Price and HMA
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='AAPL Close', linewidth=1.5, alpha=0.7)
    ax1.plot(df.index, df['HMA_50'], label='HMA 50', linewidth=1.2)
    ax1.plot(df.index, df['HMA_200'], label='HMA 200', linewidth=1.2)

    # Mark entry points
    trades_df = pd.DataFrame(portfolio.closed_trades)
    for _, trade in trades_df.iterrows():
        ax1.axvline(trade['entry_date'], color='green', alpha=0.3, linestyle='--', linewidth=0.5)

    ax1.set_title('AAPL Price and Hull Moving Averages', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Equity Curve
    ax2 = axes[1]
    equity_df = pd.DataFrame(portfolio.equity_curve)
    ax2.plot(equity_df['date'], equity_df['total_equity'], label='Total Equity', linewidth=2)
    ax2.axhline(portfolio.initial_capital, color='gray', linestyle='--', label='Initial Capital', alpha=0.5)
    ax2.set_title('Portfolio Equity Curve', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Equity ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Trade P&L
    ax3 = axes[2]
    ax3.bar(range(len(trades_df)), trades_df['pnl'],
            color=['green' if x > 0 else 'red' for x in trades_df['pnl']], alpha=0.6)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Individual Trade P&L', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('P&L ($)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(script_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    plot_file = os.path.join(reports_dir, f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {plot_file}")

    # Try to display (won't work in headless environments)
    try:
        plt.show()
    except:
        pass


if __name__ == '__main__':
    # Run the backtest
    portfolio, df = run_backtest(
        ticker='AAPL',
        start_date='2020-01-01',
        end_date='2024-10-31',
        initial_capital=100000,
        save_results=True
    )

    # Create visualizations
    print("\nGenerating visualizations...")
    try:
        plot_results(portfolio, df)
    except Exception as e:
        print(f"Could not generate plots: {e}")
        print("Results have been saved to CSV files.")

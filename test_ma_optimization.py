"""
Quick test of the MA optimization framework with minimal parameters
"""

import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(message)s"
)

from strategy_optimization_framework import (
    WalkForwardOptimizer,
    DataLoaderAdapter,
    AlpacaCryptoPortfolioEngine,
)
from ma_strategy import MovingAverageStrategy, MAStrategyParameters
from alpaca_data_loader import AlpacaDataLoader
from config_manager import Config


def quick_test(hybrid_mode: bool = False):
    """Run a minimal test of the optimization framework"""

    print("=== Quick MA Strategy Optimization Test ===\n")
    
    if hybrid_mode:
        print("Using HYBRID mode (transformed features with raw prices)\n")
    else:
        print("Using RAW OHLCV data\n")

    # Use a moderate time period for testing
    end_date = datetime.now()
    # For hybrid mode, start from when transform data is available
    if hybrid_mode:
        start_date = datetime(2024, 9, 11)  # Transform data starts Sept 2024
    else:
        start_date = end_date - timedelta(days=500)  # 500 days for raw data

    # Test parameters for 500 days
    symbol = "SOL/USD" if hybrid_mode else "BTC/USD"
    timeframe = "1h" if hybrid_mode else "1d"
    # Adjust days based on data availability
    if hybrid_mode:
        train_days = 90   # 3 months training
        test_days = 30    # 1 month testing
        step_days = 30    # 1 month step
    else:
        train_days = 180  # 6 months training
        test_days = 60    # 2 months testing
        step_days = 60    # 2 months step
    max_evaluations = 75  # Moderate evaluations

    print(f"Test Configuration:")
    print(f"  Symbol: {symbol}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Train/Test/Step: {train_days}/{test_days}/{step_days} days")
    print(f"  Max evaluations per window: {max_evaluations}")
    print()

    try:
        # Create configuration
        config = Config(
            config_path="test_config.json",
            symbol=symbol,
            timeframe=timeframe,
            min_bars=train_days * 24 + test_days * 24 + 200,  # Convert days to hours + buffer
        )

        # Create data loader
        print("1. Creating data loader...")
        base_loader = AlpacaDataLoader(config)
        data_loader = DataLoaderAdapter(
            base_loader, 
            use_transformed=False,
            hybrid_mode=hybrid_mode
        )
        if hybrid_mode:
            print("   ✓ Data loader created (hybrid mode: raw prices + transformed features)")
        else:
            print("   ✓ Data loader created (using raw data)")

        # Create strategy with narrow parameter ranges for testing
        print("2. Creating MA strategy...")
        strategy = MovingAverageStrategy(
            min_fast=12,      # 12 hours
            max_fast=48,      # 2 days
            min_slow=48,      # 2 days
            max_slow=168,     # 7 days in hours
        )
        print("   ✓ Strategy created")

        # Create optimizer
        print("3. Creating optimizer...")
        optimizer = WalkForwardOptimizer(
            strategy=strategy,
            data_loader=data_loader,
            portfolio_engine_class=AlpacaCryptoPortfolioEngine,  # Use crypto engine for BTC
            initial_balance=10000.0,  # Small balance for testing
        )
        print("   ✓ Optimizer created")

        # Run optimization
        print("\n4. Running optimization...")
        print("-" * 40)

        results = optimizer.run(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            max_evaluations=max_evaluations,
        )

        print("\n5. Results:")
        print("-" * 40)

        if results:
            print(f"   ✓ Successfully optimized {len(results)} windows")

            # Show first window details
            window, params, metrics = results[0]
            print(f"\n   First window example:")
            print(
                f"     Training: {window.train_start.date()} to {window.train_end.date()}"
            )
            print(
                f"     Testing: {window.test_start.date()} to {window.test_end.date()}"
            )
            # Cast params to correct type for better IDE support
            if isinstance(params, MAStrategyParameters):
                print(
                    f"     Best params: Fast MA = {params.fast_period}, Slow MA = {params.slow_period}"
                )
            print(f"     OOS Return: {metrics['total_return_pct']:.2f}%")
            print(f"     Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"     Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
            print(f"     Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"     Total trades: {metrics.get('total_trades', 0)}")
            print(f"     Winning trades: {metrics.get('winning_trades', 0)}")

            # Show all windows
            print(f"\n   All Windows:")
            for i, (w, p, m) in enumerate(results):
                print(
                    f"     {i+1}. Test: {w.test_start.date()} to {w.test_end.date()}, "
                    f"Return: {m['total_return_pct']:.2f}%, "
                    f"Sharpe: {m['sharpe_ratio']:.2f}, "
                    f"Calmar: {m.get('calmar_ratio', 0):.2f}, "
                    f"MaxDD: {m['max_drawdown_pct']:.2f}%, "
                    f"Trades: {m.get('total_trades', 0)}"
                )

            # Quick analysis
            analysis = optimizer.analyze_results(results)
            print(f"\n   Overall Statistics:")
            print(f"     Average OOS Return: {analysis['avg_oos_return']:.2f}%")
            print(f"     Win Rate: {analysis['win_ratio']:.1%}")
            print(f"     Avg Sharpe: {analysis['avg_sharpe']:.2f}")
            print(f"     Avg Calmar: {analysis.get('avg_calmar', 0):.2f}")
            print(f"     Avg Max Drawdown: {analysis['avg_max_drawdown']:.2f}%")

        else:
            print("   ✗ No results returned")

        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import sys
    
    # Check for hybrid flag
    hybrid_mode = "--hybrid" in sys.argv or "-h" in sys.argv
    
    success = quick_test(hybrid_mode=hybrid_mode)
    exit(0 if success else 1)

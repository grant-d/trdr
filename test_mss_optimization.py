"""
Test MSS (Market State Score) strategy optimization
"""

import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

from strategy_optimization_framework import (
    WalkForwardOptimizer, DataLoaderAdapter, AlpacaCryptoPortfolioEngine
)
from mss_strategy import SimpleMSSStrategy, MSSStrategyParameters
from alpaca_data_loader import AlpacaDataLoader
from config_manager import Config


def test_mss_optimization(hybrid_mode: bool = False):
    """Run MSS strategy optimization test"""
    
    print("=== MSS Strategy Optimization Test ===\n")
    
    if hybrid_mode:
        print("Using HYBRID mode (transformed features with raw prices)\n")
    else:
        print("Using RAW OHLCV data\n")
    
    # Use a moderate time period
    end_date = datetime.now()
    if hybrid_mode:
        start_date = datetime(2024, 9, 11)  # Transform data starts Sept 2024
    else:
        start_date = end_date - timedelta(days=500)
    
    # Test parameters
    symbol = "SOL/USD" if hybrid_mode else "BTC/USD"
    timeframe = "1h" if hybrid_mode else "1d"
    # Adjust for data availability
    if hybrid_mode:
        train_days = 90   # 3 months
        test_days = 15    # 1 month
        step_days = 15    # 1 month step
    else:
        train_days = 252   # 1 year training
        test_days = 63     # 3 months testing
        step_days = 63     # 3 months step
    max_evaluations = 100
    
    print(f"Test Configuration:")
    print(f"  Symbol: {symbol}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Train/Test/Step: {train_days}/{test_days}/{step_days} days")
    print(f"  Max evaluations per window: {max_evaluations}")
    
    try:
        # Create configuration
        config = Config(
            config_path="test_config.json",
            symbol=symbol,
            timeframe=timeframe,
            min_bars=train_days * (24 if timeframe == "1h" else 1) + test_days * (24 if timeframe == "1h" else 1) + 100
        )
        
        # Create data loader
        print("\n1. Creating data loader...")
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
        
        # Create simplified MSS strategy (7 parameters instead of 14)
        print("2. Creating Simplified MSS strategy...")
        strategy = SimpleMSSStrategy()
        print("   ✓ Strategy created")
        print("   Parameters to optimize:")
        bounds = strategy.get_parameter_bounds()
        for param, (min_val, max_val) in bounds.items():
            print(f"     - {param}: [{min_val:.2f}, {max_val:.2f}]")
        
        # Create optimizer
        print("\n3. Creating optimizer...")
        optimizer = WalkForwardOptimizer(
            strategy=strategy,
            data_loader=data_loader,
            portfolio_engine_class=AlpacaCryptoPortfolioEngine,
            initial_balance=10000.0
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
            max_evaluations=max_evaluations
        )
        
        print("\n5. Results:")
        print("-" * 40)
        
        if results:
            print(f"   ✓ Successfully optimized {len(results)} windows")
            
            # Show first window details
            window, params, metrics = results[0]
            print(f"\n   First window example:")
            print(f"     Training: {window.train_start.date()} to {window.train_end.date()}")
            print(f"     Testing: {window.test_start.date()} to {window.test_end.date()}")
            
            if isinstance(params, MSSStrategyParameters):
                print(f"\n     Best parameters:")
                print(f"       Lookback: {params.trend_lookback} days")
                print(f"       Weights: Trend={params.trend_weight:.2f}, "
                      f"Vol={params.volatility_weight:.2f}, "
                      f"Exh={params.exhaustion_weight:.2f}")
                print(f"       Thresholds: Strong=±{params.strong_bull_threshold:.1f}, "
                      f"Weak=±{params.weak_bull_threshold:.1f}")
                print(f"       ATR Multiplier: {params.atr_multiplier_strong:.2f}")
            
            print(f"\n     Performance:")
            print(f"       OOS Return: {metrics['total_return_pct']:.2f}%")
            print(f"       Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"       Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
            print(f"       Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"       Total trades: {metrics.get('total_trades', 0)}")
            print(f"       Win rate: {metrics.get('win_rate', 0):.1%}")
            
            # Show all windows
            print(f"\n   All Windows:")
            for i, (w, p, m) in enumerate(results):
                print(f"     {i+1}. Test: {w.test_start.date()} to {w.test_end.date()}, "
                      f"Return: {m['total_return_pct']:.2f}%, "
                      f"Sharpe: {m['sharpe_ratio']:.2f}, "
                      f"Calmar: {m.get('calmar_ratio', 0):.2f}, "
                      f"MaxDD: {m['max_drawdown_pct']:.2f}%, "
                      f"Trades: {m.get('total_trades', 0)}")
            
            # Analysis
            analysis = optimizer.analyze_results(results)
            print(f"\n   Overall Statistics:")
            print(f"     Average OOS Return: {analysis['avg_oos_return']:.2f}%")
            print(f"     Win Rate: {analysis['win_ratio']:.1%}")
            print(f"     Avg Sharpe: {analysis['avg_sharpe']:.2f}")
            print(f"     Avg Calmar: {analysis.get('avg_calmar', 0):.2f}")
            print(f"     Avg Max Drawdown: {analysis['avg_max_drawdown']:.2f}%")
            
            # Parameter stability analysis
            print(f"\n   Parameter Stability:")
            lookbacks = [r[1].trend_lookback for r in results]
            trend_weights = [r[1].trend_weight for r in results]
            strong_thresholds = [r[1].strong_bull_threshold for r in results]
            
            import numpy as np
            print(f"     Lookback: mean={np.mean(lookbacks):.1f}, std={np.std(lookbacks):.1f}")
            print(f"     Trend Weight: mean={np.mean(trend_weights):.2f}, std={np.std(trend_weights):.2f}")
            print(f"     Strong Threshold: mean={np.mean(strong_thresholds):.1f}, std={np.std(strong_thresholds):.1f}")
            
        else:
            print("   ✗ No results returned")
            
        print("\n✓ MSS strategy test completed successfully!")
        
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
    
    success = test_mss_optimization(hybrid_mode=hybrid_mode)
    exit(0 if success else 1)
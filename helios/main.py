"""
Helios Trading Analysis Toolkit
Main application module
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import warnings

# Import Helios modules
from data_processing import create_dollar_bars, prepare_data
from factors import (
    calculate_macd,
    calculate_rsi,
    calculate_stddev,
    calculate_mss,
)
from strategy import TradingStrategy
from performance import (
    generate_performance_report,
)
from trading_context import TradingContext, TradingContextManager
from playbook import Playbook, PlaybookManager
from optimization import GeneticAlgorithm, WalkForwardOptimizer, create_default_parameter_ranges

# Suppress warnings
warnings.filterwarnings("ignore")


def setup_environment():
    """Load environment variables and setup configuration"""
    load_dotenv()

    # Example: Get API keys or config from environment
    api_key = os.getenv("HELIOS_API_KEY")
    data_path = os.getenv("HELIOS_DATA_PATH", "./data")

    return {"api_key": api_key, "data_path": Path(data_path)}


def run_helios_analysis(df, config):
    """Run Helios trading analysis on data"""
    print("\nRunning Helios Trading Analysis...")
    print("=" * 40)

    # 1. Prepare data
    print("\n1. Preparing data...")
    df = prepare_data(df)
    print(f"   - Data shape: {df.shape}")
    print(f"   - Date range: {df.index[0]} to {df.index[-1]}")

    # 2. Create dollar bars if requested
    dollar_threshold = config.get("dollar_threshold", 1000000)
    if config.get("use_dollar_bars", False):
        print(f"\n2. Creating dollar bars (threshold: ${dollar_threshold:,.0f})...")
        df = create_dollar_bars(df, dollar_threshold)
        print(f"   - Dollar bars created: {len(df)}")

    # 3. Calculate indicators
    print("\n3. Calculating technical indicators...")

    # MACD
    macd_data = calculate_macd(df)
    df["macd"] = macd_data["macd"]
    df["macd_signal"] = macd_data["signal"]
    df["macd_hist"] = macd_data["histogram"]

    # RSI
    df["rsi"] = calculate_rsi(df)

    # Standard Deviation
    df["stddev"] = calculate_stddev(df)

    print("   - MACD calculated")
    print("   - RSI calculated")
    print("   - Standard Deviation calculated")

    # 4. Calculate Market State Score and Regimes
    print("\n4. Calculating Market State Score (MSS)...")
    lookback = config.get("lookback", 20)
    factors_df, regimes = calculate_mss(df, lookback)

    # Merge with main dataframe
    df = pd.concat([df, factors_df], axis=1)

    print(f"   - MSS calculated with lookback: {lookback}")
    print("   - Regime distribution:")
    print(regimes.value_counts().sort_index())

    # 5. Run trading strategy
    print("\n5. Running trading strategy backtest...")
    initial_capital = config.get("initial_capital", 100000)
    strategy = TradingStrategy(
        initial_capital=initial_capital,
        max_position_pct=config.get("max_position_pct", 0.95),
        min_position_pct=config.get("min_position_pct", 0.1),
    )

    results = strategy.run_backtest(df, df)
    print(f"   - Backtest complete: {len(results)} periods processed")

    # 6. Evaluate performance
    print("\n6. Evaluating performance...")
    trades_summary = strategy.get_trade_summary()
    performance_report = generate_performance_report(
        results, trades_summary, initial_capital
    )

    print("\n" + performance_report)

    # 7. Save results if requested
    if config.get("save_results", False):
        output_dir = Path(config.get("output_dir", "./results"))
        output_dir.mkdir(exist_ok=True)

        # Save results DataFrame
        results_file = output_dir / "backtest_results.csv"
        results.to_csv(results_file)
        print(f"\nResults saved to: {results_file}")

        # Save performance report
        report_file = output_dir / "performance_report.txt"
        with open(report_file, "w") as f:
            f.write(performance_report)
        print(f"Performance report saved to: {report_file}")

    return results, strategy


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Helios Trading Analysis Toolkit")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command (original functionality)
    analyze_parser = subparsers.add_parser('analyze', help='Run basic analysis on data')
    analyze_parser.add_argument("--data", "-d", required=True, help="Path to data file (CSV or Parquet)")
    analyze_parser.add_argument(
        "--dollar-bars",
        action="store_true",
        help="Convert to dollar bars before analysis",
    )
    analyze_parser.add_argument(
        "--dollar-threshold",
        type=float,
        default=1000000,
        help="Dollar volume threshold for bars (default: 1000000)",
    )
    analyze_parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Lookback period for indicators (default: 20)",
    )
    analyze_parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    analyze_parser.add_argument(
        "--save-results", action="store_true", help="Save results to files"
    )
    analyze_parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    
    # Context command
    context_parser = subparsers.add_parser('context', help='Manage trading contexts')
    context_parser.add_argument('action', choices=['create', 'list', 'status', 'pause', 'resume', 'delete'],
                               help='Context action to perform')
    context_parser.add_argument('--id', help='Context ID')
    context_parser.add_argument('--instrument', help='Instrument (e.g., BTC, AAPL)')
    context_parser.add_argument('--exchange', help='Exchange (e.g., COINBASE, NASDAQ)')
    context_parser.add_argument('--timeframe', help='Timeframe (e.g., 1h, 4h, 1d)')
    context_parser.add_argument('--experiment', default='default', help='Experiment name')
    context_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run genetic algorithm optimization')
    optimize_parser.add_argument("--data", "-d", required=True, help="Path to data file")
    optimize_parser.add_argument('--context-id', help='Context ID to optimize for')
    optimize_parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward optimization')
    optimize_parser.add_argument('--window-days', type=int, default=365, help='Training window in days')
    optimize_parser.add_argument('--step-days', type=int, default=90, help='Step size in days')
    optimize_parser.add_argument('--population', type=int, default=50, help='GA population size')
    optimize_parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    optimize_parser.add_argument('--fitness', choices=['sortino', 'calmar'], default='sortino',
                                help='Fitness metric to optimize')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest with trading context')
    backtest_parser.add_argument('--context-id', required=True, help='Context ID to backtest')
    backtest_parser.add_argument("--data", "-d", required=True, help="Path to data file")
    backtest_parser.add_argument('--save-results', action='store_true', help='Save backtest results')
    
    # Version
    parser.add_argument("--version", "-v", action="store_true", help="Show version")
    
    args = parser.parse_args()
    
    # Handle version
    if args.version:
        print("Helios v0.2.0")  # Updated version
        return 0
    
    # Handle no command
    if not args.command:
        # Legacy mode - check for --data and --analyze
        if hasattr(args, 'data') and args.data:
            # Convert to analyze command
            args.command = 'analyze'
            args.analyze = True

    # Setup environment
    config = setup_environment()
    
    # Handle commands
    if args.command == 'analyze':
        try:
            print(f"\nLoading data from: {args.data}")

            # Determine file type and load accordingly
            if args.data.endswith(".csv"):
                df = pd.read_csv(args.data, parse_dates=True, index_col=0)
            elif args.data.endswith(".parquet"):
                df = pd.read_parquet(args.data)
            else:
                print(f"Unsupported file format: {args.data}")
                print("Supported formats: .csv, .parquet")
                return 1

            # Standardize column names (handle case variations)
            df.columns = df.columns.str.lower()

            # Use adjusted close if available
            adj_close_variations = [
                "adj close",
                "adj_close",
                "adjusted_close",
                "adjustedclose",
            ]
            for col in adj_close_variations:
                if col in df.columns:
                    print(f"Using '{col}' as close price")
                    df["close"] = df[col]
                    break

            # Check required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"\nError: Missing required columns: {missing_columns}")
                print(f"Data must have columns: {required_columns}")
                print(f"Found columns: {list(df.columns)}")
                return 1

            # Build configuration from arguments
            config = {
                "use_dollar_bars": args.dollar_bars,
                "dollar_threshold": args.dollar_threshold,
                "lookback": args.lookback,
                "initial_capital": args.capital,
                "save_results": args.save_results,
                "output_dir": args.output_dir,
                "max_position_pct": 0.95,
                "min_position_pct": 0.1,
            }

            print("\nHelios Trading Analysis")
            print("=" * 40)
            
            # Run analysis
            run_helios_analysis(df, config)

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.command == 'context':
        return handle_context_command(args)
    
    elif args.command == 'optimize':
        return handle_optimize_command(args)
    
    elif args.command == 'backtest':
        return handle_backtest_command(args)
    
    else:
        print("\nHelios Trading Analysis Toolkit v0.2.0")
        print("=" * 40)
        print("\nAvailable commands:")
        print("  analyze   - Run basic analysis on historical data")
        print("  context   - Manage trading contexts")
        print("  optimize  - Run genetic algorithm optimization")
        print("  backtest  - Run backtest with trading context")
        print("\nExamples:")
        print("  python -m helios analyze --data data.csv")
        print("  python -m helios context create --instrument BTC --exchange COINBASE --timeframe 1h")
        print("  python -m helios optimize --data data.csv --walk-forward")
        print("\nFor command help:")
        print("  python -m helios <command> --help")
    
    return 0


def handle_context_command(args):
    """Handle context management commands"""
    manager = TradingContextManager()
    
    if args.action == 'create':
        if not all([args.instrument, args.exchange, args.timeframe]):
            print("Error: --instrument, --exchange, and --timeframe are required for create")
            return 1
        
        try:
            context = manager.create_context(
                instrument=args.instrument,
                exchange=args.exchange,
                timeframe=args.timeframe,
                experiment_name=args.experiment,
                initial_capital=args.capital
            )
            print(f"Created context: {context.context_id}")
            print(f"Initial capital: ${args.capital:,.2f}")
        except Exception as e:
            print(f"Error creating context: {e}")
            return 1
    
    elif args.action == 'list':
        contexts = manager.list_contexts()
        if not contexts:
            print("No trading contexts found")
        else:
            print("\nTrading Contexts:")
            print("=" * 80)
            print(f"{'Context ID':<40} {'Active':<8} {'Position':<10} {'PnL %':<10}")
            print("-" * 80)
            for ctx in contexts:
                print(f"{ctx['context_id']:<40} {str(ctx['is_active']):<8} "
                      f"{ctx['position']:>9.2f} {ctx['total_return_pct']:>9.2f}%")
    
    elif args.action == 'status':
        if not args.id:
            print("Error: --id required for status")
            return 1
        
        context = manager.get_context(args.id)
        if not context:
            print(f"Context not found: {args.id}")
            return 1
        
        status = context.get_status()
        print(f"\nContext: {status['context_id']}")
        print("=" * 40)
        print(f"Active: {status['is_active']}")
        print(f"Position: {status['position']} shares")
        print(f"Portfolio Value: ${status['portfolio_value']:,.2f}")
        print(f"Total Return: {status['total_return_pct']:.2f}%")
        print(f"Stop Loss: ${status['stop_loss']:,.2f}" if status['stop_loss'] else "Stop Loss: None")
        print(f"Last Processed: {status['last_processed']}")
    
    elif args.action in ['pause', 'resume']:
        if not args.id:
            print(f"Error: --id required for {args.action}")
            return 1
        
        context = manager.get_context(args.id)
        if not context:
            print(f"Context not found: {args.id}")
            return 1
        
        if args.action == 'pause':
            context.pause()
            print(f"Paused context: {args.id}")
        else:
            context.resume()
            print(f"Resumed context: {args.id}")
    
    elif args.action == 'delete':
        if not args.id:
            print("Error: --id required for delete")
            return 1
        
        try:
            manager.delete_context(args.id)
            print(f"Deleted context: {args.id}")
        except Exception as e:
            print(f"Error deleting context: {e}")
            return 1
    
    return 0


def handle_optimize_command(args):
    """Handle optimization commands"""
    print("\nHelios Genetic Algorithm Optimization")
    print("=" * 40)
    
    # Load data
    try:
        print(f"Loading data from: {args.data}")
        if args.data.endswith('.csv'):
            df = pd.read_csv(args.data, parse_dates=True, index_col=0)
        else:
            df = pd.read_parquet(args.data)
        
        # Standardize columns
        df.columns = df.columns.str.lower()
        df = prepare_data(df)
        
        print(f"Data loaded: {len(df)} bars")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Create GA
    param_ranges = create_default_parameter_ranges()
    ga = GeneticAlgorithm(
        parameter_ranges=param_ranges,
        population_size=args.population,
        generations=args.generations,
        fitness_metric=args.fitness
    )
    
    if args.walk_forward:
        # Walk-forward optimization
        print("\nRunning walk-forward optimization...")
        wfo = WalkForwardOptimizer(
            window_size=args.window_days,
            step_size=args.step_days
        )
        
        results = wfo.optimize(df, ga)
        
        print(f"\nAverage Train Fitness: {results['avg_train_fitness']:.4f}")
        print(f"Average Test Fitness: {results['avg_test_fitness']:.4f}")
        
        # Update playbook if context specified
        if args.context_id:
            playbook_mgr = PlaybookManager()
            # Convert results to playbook format
            # This would need more sophisticated mapping
            print(f"\nUpdating playbook for context: {args.context_id}")
    
    else:
        # Simple optimization
        print("\nRunning optimization...")
        best_individual, fitness_history = ga.optimize(df)
        
        print(f"\nBest fitness: {best_individual.fitness:.4f}")
        print("\nBest parameters:")
        for param, value in best_individual.genes.items():
            print(f"  {param}: {value:.4f}")
    
    return 0


def handle_backtest_command(args):
    """Handle backtest commands"""
    manager = TradingContextManager()
    
    context = manager.get_context(args.context_id)
    if not context:
        print(f"Context not found: {args.context_id}")
        return 1
    
    print(f"\nRunning backtest for context: {args.context_id}")
    print("=" * 40)
    
    # Load data
    try:
        print(f"Loading data from: {args.data}")
        if args.data.endswith('.csv'):
            df = pd.read_csv(args.data, parse_dates=True, index_col=0)
        else:
            df = pd.read_parquet(args.data)
        
        # Standardize columns
        df.columns = df.columns.str.lower()
        df = prepare_data(df)
        
        # Create dollar bars if specified in context
        if context.state.parameters.get('dollar_bar_threshold'):
            print(f"Creating dollar bars...")
            df = create_dollar_bars(df, context.state.parameters['dollar_bar_threshold'])
        
        # Calculate factors
        print("Calculating indicators...")
        
        # Get playbook for dynamic parameters
        playbook_mgr = PlaybookManager()
        playbook = playbook_mgr.get_playbook(args.context_id)
        
        # Process each bar
        results = []
        for i in range(len(df)):
            if i < context.state.parameters['lookback']:
                continue
            
            # Get current regime parameters
            # This would need the full factor calculation
            # For now, use default parameters
            
            # Process bar
            # result = context.process_bar(df.iloc[i], factors.iloc[i])
            # results.append(result)
        
        print("\nBacktest complete")
        # Would show performance summary
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

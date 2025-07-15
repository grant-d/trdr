"""
Helios Trading Analysis Toolkit
Main application module
"""

import os
import sys
import argparse
import pandas as pd
from generic_algorithm import estimate_asset_volatility_scale
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import warnings
from chalk import black
import json

# Import Helios modules
from data_processing import create_dollar_bars, prepare_data
from factors import (
    calculate_macd,
    calculate_rsi,
    calculate_stddev,
    calculate_mss,
)
from strategy_enhanced import EnhancedTradingStrategy
from performance import (
    generate_performance_report,
)
from trading_context import TradingContext, TradingContextManager
from playbook import Playbook, PlaybookManager
from optimization import GeneticAlgorithm, WalkForwardOptimizer, auto_detect_dollar_thresholds, create_enhanced_parameter_ranges
from args_types import OptimizeArgs, TestArgs, AnalysisConfig
from typing import Union

# Suppress warnings
warnings.filterwarnings("ignore")


def setup_environment():
    """Load environment variables and setup configuration"""
    load_dotenv()

    # Example: Get API keys or config from environment
    api_key = os.getenv("HELIOS_API_KEY")
    data_path = os.getenv("HELIOS_DATA_PATH", "./data")

    return {"api_key": api_key, "data_path": Path(data_path)}


def play_completion_sound():
    """Play a sound notification when process completes"""
    try:
        import os

        os.system("afplay /System/Library/Sounds/Glass.aiff")  # macOS
    except:
        try:
            print("\a")  # Terminal bell as fallback
        except:
            pass


def run_helios_analysis(df: pd.DataFrame, config: Union[dict, AnalysisConfig]) -> tuple:
    """Run Helios trading analysis on data"""
    print("\nRunning Helios Trading Analysis...")
    print("=" * 40)

    # 1. Prepare data
    print("\n1. Preparing data...")
    df = prepare_data(df)
    print(f"   - Data shape: {df.shape}")
    print(f"   - Date range: {df.index[0]} to {df.index[-1]}")

    # 2. Create dollar bars
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
    df["macd_hist_norm"] = macd_data["histogram_normalized"]

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
    strategy = EnhancedTradingStrategy(
        initial_capital=initial_capital,
        max_position_fraction=config.get("max_position_pct", 0.95),
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
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command (original functionality)
    analyze_parser = subparsers.add_parser("analyze", help="Run basic analysis on data")
    analyze_parser.add_argument(
        "--data", "-d", required=True, help="Path to data file (CSV or Parquet)"
    )
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
    context_parser = subparsers.add_parser("context", help="Manage trading contexts")
    context_parser.add_argument(
        "action",
        choices=["create", "list", "status", "pause", "resume", "delete"],
        help="Context action to perform",
    )
    context_parser.add_argument("--id", help="Context ID")
    context_parser.add_argument("--instrument", help="Instrument (e.g., BTC, AAPL)")
    context_parser.add_argument("--exchange", help="Exchange (e.g., COINBASE, NASDAQ)")
    context_parser.add_argument("--timeframe", help="Timeframe (e.g., 1h, 4h, 1d)")
    context_parser.add_argument(
        "--experiment", default="default", help="Experiment name"
    )
    context_parser.add_argument(
        "--capital", type=float, default=100000, help="Initial capital"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize", help="Run genetic algorithm optimization"
    )
    optimize_parser.add_argument(
        "--data", "-d", required=True, help="Path to data file"
    )
    optimize_parser.add_argument(
        "--dollar-threshold",
        default="auto",
        help='Dollar volume threshold for bars (default: auto-detect, number, or "none" to disable dollar bars)',
    )
    optimize_parser.add_argument("--context-id", help="Context ID to optimize for")
    optimize_parser.add_argument(
        "--population", type=int, default=50, help="GA population size"
    )
    optimize_parser.add_argument(
        "--generations", type=int, default=20, help="Number of generations"
    )
    optimize_parser.add_argument(
        "--fitness",
        choices=["sortino", "calmar"],
        default="sortino",
        help="Fitness metric to optimize",
    )
    optimize_parser.add_argument(
        "--allow-shorts",
        action="store_true",
        help="Allow short positions (disabled by default)",
    )
    optimize_parser.add_argument(
        "--output",
        nargs="?",
        const="",
        default=None,
        help="Save results (use alone for params.json, or with prefix like --output msft for msft-params.json)",
    )
    optimize_parser.add_argument(
        "--output-dir",
        default="./optimization_results",
        help="Output directory for optimization results (default: ./optimization_results)",
    )
    optimize_parser.add_argument(
        "--test", action="store_true", help="Run test immediately after optimization"
    )
    optimize_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after optimization (requires --test)",
    )
    optimize_parser.add_argument(
        "--llm", action="store_true", help="Generate LLM analysis report after optimization (requires --test)"
    )

    # Test command (formerly run-optimized)
    test_parser = subparsers.add_parser(
        "test", help="Test strategy with saved optimization parameters"
    )
    test_parser.add_argument(
        "--data",
        "-d",
        help="Path to data file (optional - uses saved path from optimization if not provided)",
    )
    test_parser.add_argument(
        "--params", "-p", required=True, help="Path to optimized parameters JSON file"
    )
    test_parser.add_argument(
        "--dollar-threshold",
        default=None,
        help='Dollar volume threshold for bars (number, "auto", or "none" to disable - overrides saved setting if specified)',
    )
    test_parser.add_argument(
        "--capital", type=float, default=100000, help="Initial capital"
    )
    test_parser.add_argument(
        "--save-results", action="store_true", help="Save test results"
    )
    test_parser.add_argument(
        "--output-dir",
        default="./strategy_results",
        help="Output directory for results",
    )
    test_parser.add_argument(
        "--allow-shorts",
        action="store_true",
        help="Allow short positions (disabled by default)",
    )
    test_parser.add_argument(
        "--plot", action="store_true", help="Show performance plots"
    )
    test_parser.add_argument(
        "--llm", action="store_true", help="Generate LLM analysis report with trade data and market context"
    )

    # Backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", help="Run backtest with trading context"
    )
    backtest_parser.add_argument(
        "--context-id", required=True, help="Context ID to backtest"
    )
    backtest_parser.add_argument(
        "--data", "-d", required=True, help="Path to data file"
    )
    backtest_parser.add_argument(
        "--save-results", action="store_true", help="Save backtest results"
    )

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
        if hasattr(args, "data") and args.data:
            # Convert to analyze command
            args.command = "analyze"
            args.analyze = True

    # Setup environment
    config = setup_environment()

    # Handle commands
    if args.command == "analyze":
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
            config = AnalysisConfig(
                use_dollar_bars=args.dollar_bars,
                dollar_threshold=args.dollar_threshold,
                lookback=args.lookback,
                initial_capital=args.capital,
                save_results=args.save_results,
                output_dir=args.output_dir,
                max_position_pct=0.95,
                min_position_pct=0.1
            )

            print("\nHelios Trading Analysis")
            print("=" * 40)

            # Run analysis
            run_helios_analysis(df, config)

        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            return 1

    elif args.command == "context":
        return handle_context_command(args)

    elif args.command == "optimize":
        return handle_optimize_command(args)

    elif args.command == "test":
        return handle_test_command(args)

    elif args.command == "backtest":
        return handle_backtest_command(args)

    else:
        print("\nHelios Trading Analysis Toolkit v0.2.0")
        print("=" * 40)
        print("\nAvailable commands:")
        print("  analyze      - Run basic analysis on historical data")
        print("  optimize     - Run genetic algorithm optimization")
        print("  test         - Test strategy with saved optimization parameters")
        print("  context      - Manage trading contexts")
        print("  backtest     - Run backtest with trading context")
        print("\nExamples:")
        print("  python -m helios analyze --data data.csv --dollar-bars")
        print(
            "  python -m helios optimize --data data.csv --population 50 --generations 20"
        )
        print(
            "  python -m helios optimize --data data.csv --dollar-threshold none --population 50"
        )
        print(
            "  python -m helios test --params optimization_results/optimized_parameters.json"
        )
        print(
            "  python -m helios test --params optimization_results/optimized_parameters.json --plot"
        )
        print(
            "  python -m helios test --data new_data.csv --params optimization_results/optimized_parameters.json"
        )
        print(
            "  python -m helios context create --instrument BTC --exchange COINBASE --timeframe 1h"
        )
        print("\nFor command help:")
        print("  python -m helios <command> --help")

    return 0


def handle_context_command(args):
    """Handle context management commands"""
    manager = TradingContextManager()

    if args.action == "create":
        if not all([args.instrument, args.exchange, args.timeframe]):
            print(
                "Error: --instrument, --exchange, and --timeframe are required for create"
            )
            return 1

        try:
            context = manager.create_context(
                instrument=args.instrument,
                exchange=args.exchange,
                timeframe=args.timeframe,
                experiment_name=args.experiment,
                initial_capital=args.capital,
            )
            print(f"Created context: {context.context_id}")
            print(f"Initial capital: ${args.capital:,.2f}")
        except Exception as e:
            print(f"Error creating context: {e}")
            return 1

    elif args.action == "list":
        contexts = manager.list_contexts()
        if not contexts:
            print("No trading contexts found")
        else:
            print("\nTrading Contexts:")
            print("=" * 80)
            print(f"{'Context ID':<40} {'Active':<8} {'Position':<10} {'PnL %':<10}")
            print("-" * 80)
            for ctx in contexts:
                print(
                    f"{ctx['context_id']:<40} {str(ctx['is_active']):<8} "
                    f"{ctx['position']:>9.2f} {ctx['total_return_pct']:>9.2f}%"
                )

    elif args.action == "status":
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
        print(
            f"Stop Loss: ${status['stop_loss']:,.2f}"
            if status["stop_loss"]
            else "Stop Loss: None"
        )
        print(f"Last Processed: {status['last_processed']}")

    elif args.action in ["pause", "resume"]:
        if not args.id:
            print(f"Error: --id required for {args.action}")
            return 1

        context = manager.get_context(args.id)
        if not context:
            print(f"Context not found: {args.id}")
            return 1

        if args.action == "pause":
            context.pause()
            print(f"Paused context: {args.id}")
        else:
            context.resume()
            print(f"Resumed context: {args.id}")

    elif args.action == "delete":
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


def handle_optimize_command(args: Union[argparse.Namespace, OptimizeArgs]) -> int:
    """Handle optimization commands
    
    Parameters
    ----------
    args : Union[argparse.Namespace, OptimizeArgs]
        Optimization command arguments
        
    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    print("\nHelios Genetic Algorithm Optimization")
    print("=" * 40)

    # Load data
    try:
        print(f"Loading data from: {args.data}")
        if args.data.endswith(".csv"):
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

    # Extract symbol and timeframe for fitness history
    data_filename = Path(args.data).stem  # Get filename without extension
    # Try to extract symbol from filename (common patterns)
    symbol = "UNKNOWN"
    if "_" in data_filename:
        parts = data_filename.split("_")
        if len(parts) >= 2:
            symbol = f"{parts[0]}_{parts[1]}"  # e.g., BTC_USD from BTC_USD_1min.csv

    # Parse dollar threshold argument
    dollar_threshold_arg = args.dollar_threshold or "auto"
    dollar_threshold = None

    try:
        dollar_threshold = float(dollar_threshold_arg)
        print(f"Using specified dollar threshold: ${dollar_threshold:,.0f}")
    except (ValueError, TypeError):
        print(f"Warning: Invalid dollar threshold '{dollar_threshold_arg}', using auto-detect")
        dollar_threshold = "auto"

    # Auto-detect threshold
    if dollar_threshold == "auto":
        # Use 5 walk-forward windows by default for robust optimization
        dollar_threshold = auto_detect_dollar_thresholds(df, walk_forward_windows=5, symbol=symbol)
        # Create dollar bars immediately to get count

    df = create_dollar_bars(df, dollar_threshold)
    print(f"  {black('Dollar bar threshold:'.ljust(22))} ${dollar_threshold:,.0f}")
    print(f"  {black('Dollar bars created:'.ljust(22))} {len(df)}")

    # Estimate volatility scale for adaptive regime thresholds
    volatility_scale = estimate_asset_volatility_scale(df)
    param_ranges = create_enhanced_parameter_ranges(volatility_scale)
    print("Using enhanced strategy with gradual entries and adaptive regime thresholds")

    # Determine timeframe string
    timeframe = f"dollar_{int(dollar_threshold)}"

    ga = GeneticAlgorithm(
        parameter_config=param_ranges,
        symbol=symbol,
        timeframe=timeframe,
        population_size=args.population,
        generations=args.generations,
        allow_shorts=args.allow_shorts,
    )

    # Walk-forward optimization
    print("\nRunning walk-forward optimization...")

    # Calculate number of windows based on available bars
    # For dollar bars, we use bar count not time-based windows
    total_bars = len(df)
    # With 70/30 split, we need ~333 bars per window to get 100 bars in test set
    min_bars_per_window = 333  # Ensures ~100 bars in test set with 70/30 split
    
    # Calculate how many windows we can actually support with the minimum requirement
    max_possible_windows = total_bars // min_bars_per_window
    
    if max_possible_windows >= 1:
        # We have enough data for at least one proper window
        n_windows = min(max_possible_windows, 5)  # Use up to 5 windows if possible
    else:
        # Not enough data for even one proper window
        # Use just 1 window with whatever data we have
        n_windows = 1
        print(f"⚠️  Warning: Only {total_bars} bars available (need {min_bars_per_window} per window)")
        print(f"   Using 1 window with limited data")
    
    bars_per_window = total_bars // n_windows
    test_bars = int(bars_per_window * 0.3)
    
    print(f"  Total bars: {total_bars}")
    print(f"  Windows: {n_windows} (each with {bars_per_window} bars, ~{test_bars} in test)")

    wfo = WalkForwardOptimizer(
        n_windows=n_windows, 
        train_ratio=0.7  # Default 70/30 train/test split
    )

    results = wfo.optimize(df, ga, save_results=True, output_dir=args.output_dir)

    # Extract best parameters from walk-forward results
    best_test_fitness = -float('inf')
    best_individual = None

    for window_result in results['windows']:
        if window_result['test_fitness'] > best_test_fitness:
            best_test_fitness = window_result['test_fitness']
            # Recreate Individual from saved genes
            from generic_algorithm import Individual
            best_individual = Individual(ga.parameter_ranges)
            best_individual.genes = window_result['best_parameters']
            best_individual.fitness = window_result['train_fitness']

    if best_individual is None:
        print("Error: No valid parameters found in walk-forward optimization")
        return 1

    print(f"\nAverage Train Fitness: {results['avg_train_fitness']:.4f}")
    print(f"Average Test Fitness: {results['avg_test_fitness']:.4f}")
    print(f"\nBest fitness: {best_individual.fitness:.4f}")
    print("Best parameters:")
    for param, value in best_individual.genes.items():
        print(f"  {param}: {value:.4f}")

    # Sound notification when walk-forward optimization completes
    play_completion_sound()

    # Save results if output specified
    if args.output is not None:
        print("\nSaving optimization results...")
        results_dir = Path(args.output_dir)
        results_dir.mkdir(exist_ok=True)

        # Save optimized parameters
        # Convert data path to absolute path for portability
        data_path = Path(args.data).resolve()

        params_data = {
            "parameters": {k: float(v) for k, v in best_individual.genes.items()},
            "fitness": float(best_individual.fitness),
            "allow_shorts": args.allow_shorts,
            "data_file": str(data_path),  # Save absolute path
            "dollar_threshold": dollar_threshold,  # None if disabled, number if enabled
            "population_size": args.population,
            "generations": args.generations,
            "parameter_ranges": {k: v.to_dict() for k, v in param_ranges.items()},
            "walk_forward_results": {
                "avg_train_fitness": results['avg_train_fitness'],
                "avg_test_fitness": results['avg_test_fitness'],
                "best_test_fitness": float(best_test_fitness),
                "n_windows": n_windows
            }
        }

        # Use output prefix if provided
        if args.output and args.output.strip():  # If output has a value
            prefix = args.output.strip()
            params_filename = f"{prefix}-params.json"
        else:  # If output is empty string or just flag
            params_filename = "params.json"

        params_file = results_dir / params_filename
        with open(params_file, "w") as f:
            json.dump(params_data, f, indent=2)
        print(f"   Optimized parameters saved to: {params_file}")

        # Run test if requested
        if args.test:
            print("\n" + "=" * 60)
            print("Running test with optimized parameters...")
            print("=" * 60)

            # Import TestArgs properly
            from args_types import TestArgs
            
            # Create a proper TestArgs instance
            test_args = TestArgs(
                params=str(params_file),
                data=None,  # Use saved data path
                dollar_threshold=None,  # Use saved threshold
                capital=100000,
                save_results=False,  # Default for test command
                output_dir="./strategy_results",
                allow_shorts=False,  # Use saved setting
                plot=args.plot,  # Pass through plot flag
                llm=args.llm  # Pass through llm flag
            )

            # Run the test
            result = handle_test_command(test_args)

            if result != 0:
                print("Warning: Test command failed")
            elif args.plot:
                print("\nPlots have been generated (if test was successful)")

    # Update playbook if context specified
    if args.context_id:
        playbook_mgr = PlaybookManager()
        # Convert results to playbook format
        # This would need more sophisticated mapping
        print(f"\nUpdating playbook for context: {args.context_id}")

    return 0


def handle_test_command(args: Union[argparse.Namespace, TestArgs]) -> int:
    """Handle test commands
    
    Parameters
    ----------
    args : Union[argparse.Namespace, TestArgs]
        Test command arguments
        
    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    print("\nHelios Strategy Testing")
    print("=" * 40)

    # Load optimized parameters
    try:
        print(f"Loading parameters from: {args.params}")
        import json

        with open(args.params, "r") as f:
            params_data = json.load(f)

        opt_params = params_data["parameters"]
        allow_shorts = args.allow_shorts or params_data.get("allow_shorts", False)
        dollar_threshold = params_data.get("dollar_threshold")
        use_dollar_bars = dollar_threshold is not None

        print("Loaded strategy parameters:")
        for param, value in sorted(opt_params.items()):
            print(f"  {param}: {value:.4f}")
        print(f"Original fitness: {params_data.get('fitness', 'N/A')}")
        print(f"Allow shorts: {allow_shorts}")
        print(f"Use dollar bars: {use_dollar_bars}")
        if use_dollar_bars and dollar_threshold:
            print(f"Dollar threshold: ${dollar_threshold:,.0f}")

    except Exception as e:
        print(f"Error loading parameters: {e}")
        return 1

    # Determine data file path
    if args.data:
        data_file = args.data
    else:
        # Use saved data file from optimization
        data_file = params_data.get("data_file")
        if not data_file:
            print("Error: No data file specified and none found in parameters file")
            return 1
        print(f"Using data file from optimization: {data_file}")

    # Load data
    try:
        print(f"\nLoading data from: {data_file}")
        if data_file.endswith(".csv"):
            df = pd.read_csv(data_file, parse_dates=True, index_col=0)
        else:
            df = pd.read_parquet(data_file)

        # Standardize columns
        df.columns = df.columns.str.lower()

        # Handle adjusted close
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

        df = prepare_data(df)
        print(f"Data loaded: {len(df)} bars")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Extract symbol from data filename for caching
    data_filename = Path(data_file).stem
    symbol = "UNKNOWN"
    if "_" in data_filename:
        parts = data_filename.split("_")
        if len(parts) >= 2:
            symbol = f"{parts[0]}_{parts[1]}"

    # Parse dollar threshold override if provided
    if args.dollar_threshold is not None:
        if args.dollar_threshold in ["none", "off", "disable", "false"]:
            use_dollar_bars = False
            dollar_threshold = None
            print("Command line override: Dollar bars disabled")
        elif args.dollar_threshold == "auto":
            use_dollar_bars = True
            from optimization import auto_detect_dollar_thresholds

            dollar_threshold = auto_detect_dollar_thresholds(df, walk_forward_windows=5, symbol=symbol)
            print(
                f"Command line override: Auto-detected dollar threshold: ${dollar_threshold:,.0f}"
            )
        else:
            try:
                use_dollar_bars = True
                dollar_threshold = float(args.dollar_threshold)
                print(
                    f"Command line override: Using dollar threshold: ${dollar_threshold:,.0f}"
                )
            except ValueError:
                print(
                    f"Warning: Invalid dollar threshold '{args.dollar_threshold}', using saved setting"
                )

    # Convert to dollar bars if enabled
    if use_dollar_bars and dollar_threshold is not None:
        print(f"\nCreating dollar bars (threshold: ${dollar_threshold:,.0f})...")
        df = create_dollar_bars(df, dollar_threshold)
        print(f"Dollar bars created: {len(df)}")
    else:
        print(f"\nUsing traditional candle data ({len(df)} bars)")

    # Calculate factors with optimized parameters
    print("\nCalculating indicators...")
    weights = {
        "trend": opt_params.get("weight_trend", 0.4),
        "volatility": opt_params.get("weight_volatility", 0.3),
        "exhaustion": opt_params.get("weight_exhaustion", 0.3),
    }
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    lookback = int(opt_params.get("lookback_int", 20))
    factors_df, regimes = calculate_mss(df, lookback, weights)

    # Add indicators
    macd_data = calculate_macd(df)
    factors_df["macd_hist"] = macd_data["histogram"]
    factors_df["rsi"] = calculate_rsi(df)

    # Merge data
    combined_df = pd.concat([df, factors_df], axis=1)

    # Always use enhanced strategy with dollar bars
    print(f"\nRunning backtest with enhanced strategy...")

    from strategy_enhanced import EnhancedTradingStrategy

    strategy = EnhancedTradingStrategy(
        initial_capital=args.capital,
        max_position_fraction=opt_params.get("max_position_pct", 1.0),
        entry_step_size=opt_params.get("entry_step_size", 0.2),
        stop_loss_multiplier_strong=opt_params.get("stop_loss_multiplier_strong", 2.0),
        stop_loss_multiplier_weak=opt_params.get("stop_loss_multiplier_weak", 1.0),
        strong_bull_threshold=opt_params.get("strong_bull_threshold", 50.0),
        weak_bull_threshold=opt_params.get("weak_bull_threshold", 20.0),
        neutral_upper=opt_params.get("neutral_threshold_upper", 20.0),
        neutral_lower=opt_params.get("neutral_threshold_lower", -20.0),
        weak_bear_threshold=opt_params.get("weak_bear_threshold", -20.0),
        strong_bear_threshold=opt_params.get("strong_bear_threshold", -50.0),
        allow_shorts=allow_shorts,
    )

    # Run backtest
    results = strategy.run_backtest(combined_df, combined_df)
    trades_summary = strategy.get_trade_summary()

    # Debug: Check if results DataFrame has required columns
    if results.empty:
        print("Warning: Backtest returned empty results")
        return 1
        
    if 'portfolio_value' not in results.columns:
        print(f"Warning: Backtest results missing 'portfolio_value' column")
        print(f"Available columns: {list(results.columns)}")
        return 1

    # Generate performance report
    performance_report = generate_performance_report(
        results, trades_summary, args.capital
    )
    print("\n" + performance_report)
    
    # Generate LLM analysis report if requested
    if args.llm:
        generate_llm_analysis_report(
            results, strategy, combined_df, opt_params, performance_report, 
            data_file, args.capital, dollar_threshold, use_dollar_bars
        )

    # Show plots if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            print("\nGenerating performance plots...")

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Strategy Performance Analysis", fontsize=16)

            # 1. Equity Curve
            portfolio_values = results["portfolio_value"]
            ax1.plot(
                portfolio_values.index,
                portfolio_values.values,
                "b-",
                linewidth=2,
                label="Portfolio Value",
            )
            ax1.axhline(
                y=args.capital,
                color="gray",
                linestyle="--",
                alpha=0.7,
                label="Initial Capital",
            )
            ax1.set_title("Equity Curve")
            ax1.set_ylabel("Portfolio Value ($)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Drawdown
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
            ax2.plot(drawdown.index, drawdown.values, "r-", linewidth=1)
            ax2.set_title("Drawdown (%)")
            ax2.set_ylabel("Drawdown (%)")
            ax2.grid(True, alpha=0.3)

            # 3. Returns Distribution
            returns = portfolio_values.pct_change().dropna()
            ax3.hist(
                returns * 100, bins=50, alpha=0.7, color="green", edgecolor="black"
            )
            ax3.axvline(
                returns.mean() * 100,
                color="red",
                linestyle="--",
                label=f"Mean: {returns.mean()*100:.2f}%",
            )
            ax3.set_title("Returns Distribution")
            ax3.set_xlabel("Returns (%)")
            ax3.set_ylabel("Frequency")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. Rolling Sharpe Ratio (30-period)
            if len(returns) > 30:
                rolling_sharpe = (
                    returns.rolling(30).mean()
                    / returns.rolling(30).std()
                    * np.sqrt(252)
                )
                ax4.plot(
                    rolling_sharpe.index, rolling_sharpe.values, "purple", linewidth=2
                )
                ax4.axhline(
                    y=1.0, color="gray", linestyle="--", alpha=0.7, label="Sharpe = 1.0"
                )
                ax4.set_title("Rolling Sharpe Ratio (30-period)")
                ax4.set_ylabel("Sharpe Ratio")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "Insufficient data\nfor rolling Sharpe",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
                ax4.set_title("Rolling Sharpe Ratio (Insufficient Data)")

            # Format x-axes for dates
            for ax in [ax1, ax2, ax4]:
                if len(portfolio_values) > 0:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\nMatplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"\nError generating plots: {e}")

    # Save results if requested
    if args.save_results:
        print("\nSaving results...")
        results_dir = Path(args.output_dir)
        results_dir.mkdir(exist_ok=True)

        # Save backtest results
        results_file = results_dir / "backtest_results.csv"
        results.to_csv(results_file)
        print(f"   Backtest results saved to: {results_file}")

        # Save performance report
        report_file = results_dir / "performance_report.txt"
        with open(report_file, "w") as f:
            f.write(performance_report)
        print(f"   Performance report saved to: {report_file}")

        # Save trades log (always enhanced strategy now)
        if hasattr(strategy, "trades"):
            trades_file = results_dir / "trades_log.csv"
            trades_df = pd.DataFrame(
                [
                    {
                        "timestamp": t.timestamp,
                        "action": t.action,
                        "units": t.units,
                        "price": t.price,
                        "pnl": t.pnl,
                        "reason": t.reason,
                        "portfolio_value": t.portfolio_value,
                    }
                    for t in strategy.trades
                ]
            )
            trades_df.to_csv(trades_file, index=False)
            print(f"   Trades log saved to: {trades_file}")

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
        if args.data.endswith(".csv"):
            df = pd.read_csv(args.data, parse_dates=True, index_col=0)
        else:
            df = pd.read_parquet(args.data)

        # Standardize columns
        df.columns = df.columns.str.lower()
        df = prepare_data(df)

        # Create dollar bars if specified in context
        if context.state.parameters.get("dollar_bar_threshold"):
            print(f"Creating dollar bars...")
            df = create_dollar_bars(
                df, context.state.parameters["dollar_bar_threshold"]
            )

        # Calculate factors
        print("Calculating indicators...")

        # Get playbook for dynamic parameters
        playbook_mgr = PlaybookManager()
        playbook = playbook_mgr.get_playbook(args.context_id)

        # Process each bar
        results = []
        for i in range(len(df)):
            if i < context.state.parameters["lookback"]:
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


def generate_llm_analysis_report(results, strategy, combined_df, opt_params, performance_report, 
                                data_file, initial_capital, dollar_threshold, use_dollar_bars):
    """Generate comprehensive LLM analysis report for performance review"""
    import json
    from datetime import datetime
    
    # Create output directory
    results_dir = Path("./llm_analysis")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"llm_analysis_report_{timestamp}.md"
    
    # Calculate additional metrics for analysis
    portfolio_values = results['portfolio_value']
    returns = portfolio_values.pct_change().dropna()
    
    # Analyze trades by market regime
    trades_by_regime = {}
    regime_performance = {}
    
    if hasattr(strategy, 'trades') and len(strategy.trades) > 0:
        for trade in strategy.trades:
            # Find the regime at trade time
            trade_idx = combined_df.index.get_indexer([trade.timestamp], method='nearest')[0]
            if trade_idx >= 0 and trade_idx < len(combined_df):
                mss_score = combined_df.iloc[trade_idx].get('mss_score', 0)
                
                # Determine regime based on thresholds
                if mss_score >= opt_params.get('strong_bull_threshold', 50):
                    regime = 'Strong Bull'
                elif mss_score >= opt_params.get('weak_bull_threshold', 20):
                    regime = 'Weak Bull'
                elif mss_score >= opt_params.get('neutral_threshold_upper', 20):
                    regime = 'Neutral Upper'
                elif mss_score >= opt_params.get('neutral_threshold_lower', -20):
                    regime = 'Neutral Lower'
                elif mss_score >= opt_params.get('weak_bear_threshold', -20):
                    regime = 'Weak Bear'
                else:
                    regime = 'Strong Bear'
                
                if regime not in trades_by_regime:
                    trades_by_regime[regime] = []
                    regime_performance[regime] = {'total_pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0}
                
                trades_by_regime[regime].append(trade)
                regime_performance[regime]['total_pnl'] += trade.pnl if trade.pnl else 0
                regime_performance[regime]['trades'] += 1
                if trade.pnl and trade.pnl > 0:
                    regime_performance[regime]['wins'] += 1
                else:
                    regime_performance[regime]['losses'] += 1
    
    # Analyze volatility periods
    if 'atr' in combined_df.columns:
        volatility_periods = combined_df['atr'].rolling(20).mean()
        high_vol_threshold = volatility_periods.quantile(0.75)
        low_vol_threshold = volatility_periods.quantile(0.25)
    else:
        volatility_periods = returns.rolling(20).std()
        high_vol_threshold = volatility_periods.quantile(0.75)
        low_vol_threshold = volatility_periods.quantile(0.25)
    
    # Identify drawdown periods
    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak
    max_drawdown_idx = drawdown.idxmin()
    max_drawdown_value = drawdown.min()
    
    # Calculate Sortino ratio
    negative_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() / negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else None
    sortino_str = f"{sortino_ratio:.2f}" if sortino_ratio is not None else "N/A"
    
    # Generate the LLM analysis report
    report_content = f"""# Helios Trading Strategy - LLM Performance Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## EXECUTIVE SUMMARY
{performance_report}

## STRATEGY CONFIGURATION
### Data Source
- **File**: {data_file}
- **Initial Capital**: ${initial_capital:,.2f}
- **Dollar Bars**: {'Enabled' if use_dollar_bars else 'Disabled'}
- **Dollar Threshold**: {"${:,.0f}".format(dollar_threshold) if use_dollar_bars and dollar_threshold else "N/A"}

### Optimized Parameters
```json
{json.dumps(opt_params, indent=2)}
```

### Market Regime Thresholds
- **Strong Bull**: MSS ≥ {opt_params.get('strong_bull_threshold', 50):.1f}
- **Weak Bull**: {opt_params.get('weak_bull_threshold', 20):.1f} ≤ MSS < {opt_params.get('strong_bull_threshold', 50):.1f}
- **Neutral Upper**: {opt_params.get('neutral_threshold_upper', 20):.1f} ≤ MSS < {opt_params.get('weak_bull_threshold', 20):.1f}
- **Neutral Lower**: {opt_params.get('neutral_threshold_lower', -20):.1f} ≤ MSS < {opt_params.get('neutral_threshold_upper', 20):.1f}
- **Weak Bear**: {opt_params.get('weak_bear_threshold', -20):.1f} ≤ MSS < {opt_params.get('neutral_threshold_lower', -20):.1f}
- **Strong Bear**: MSS < {opt_params.get('strong_bear_threshold', -50):.1f}

## REGIME-BASED PERFORMANCE ANALYSIS

### Trading Performance by Market Regime
"""

    for regime, perf in regime_performance.items():
        if perf['trades'] > 0:
            win_rate = (perf['wins'] / perf['trades']) * 100
            avg_pnl = perf['total_pnl'] / perf['trades']
            report_content += f"""
#### {regime}
- **Total Trades**: {perf['trades']}
- **Win Rate**: {win_rate:.1f}%
- **Total P&L**: ${perf['total_pnl']:,.2f}
- **Average P&L per Trade**: ${avg_pnl:.2f}
- **Wins**: {perf['wins']}, **Losses**: {perf['losses']}
"""

    report_content += f"""

## DETAILED TRADE LOG ANALYSIS

### Sample Recent Trades (Last 20)
"""

    if hasattr(strategy, 'trades') and len(strategy.trades) > 0:
        recent_trades = strategy.trades[-20:] if len(strategy.trades) >= 20 else strategy.trades
        report_content += "| Timestamp | Action | Units | Price | P&L | Reason | Portfolio Value |\n"
        report_content += "|-----------|--------|-------|-------|-----|--------|----------------|\n"
        
        for trade in recent_trades:
            timestamp = trade.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(trade.timestamp, 'strftime') else str(trade.timestamp)
            pnl_str = f"${trade.pnl:.2f}" if trade.pnl else "N/A"
            portfolio_str = f"${trade.portfolio_value:,.2f}" if trade.portfolio_value else "N/A"
            report_content += f"| {timestamp} | {trade.action} | {trade.units:.2f} | ${trade.price:.2f} | {pnl_str} | {trade.reason} | {portfolio_str} |\n"

    report_content += f"""

## MARKET CONDITIONS ANALYSIS

### Volatility Analysis
- **High Volatility Threshold**: {high_vol_threshold:.4f}
- **Low Volatility Threshold**: {low_vol_threshold:.4f}
- **Current Volatility Regime**: {"High" if volatility_periods.iloc[-1] > high_vol_threshold else "Low" if volatility_periods.iloc[-1] < low_vol_threshold else "Medium"}

### Maximum Drawdown Event
- **Date**: {max_drawdown_idx}
- **Drawdown**: {max_drawdown_value:.2%}
- **Portfolio Value at Drawdown**: ${portfolio_values.loc[max_drawdown_idx]:,.2f}

## KEY PERFORMANCE INDICATORS

### Return Metrics
- **Total Return**: {((portfolio_values.iloc[-1] / initial_capital) - 1) * 100:.2f}%
- **Annualized Return**: {(((portfolio_values.iloc[-1] / initial_capital) ** (252 / len(portfolio_values))) - 1) * 100:.2f}%
- **Volatility (Annualized)**: {returns.std() * np.sqrt(252) * 100:.2f}%

### Risk Metrics
- **Maximum Drawdown**: {max_drawdown_value:.2%}
- **Sharpe Ratio**: {(returns.mean() / returns.std() * np.sqrt(252)):.2f}
- **Sortino Ratio**: {sortino_str}

## PROMPTS FOR LLM ANALYSIS

### Performance Review Questions
1. **Regime Analysis**: Which market regime shows the worst performance? What specific pattern in the trade log suggests the strategy is struggling in that regime?

2. **Volatility Impact**: How does the strategy perform during high vs. low volatility periods? Are there specific volatility conditions where the strategy consistently loses money?

3. **Drawdown Analysis**: What market conditions led to the maximum drawdown period? What regime and volatility characteristics were present?

4. **Trade Timing**: Looking at the recent trades, are there patterns in entry/exit timing that suggest systematic issues?

5. **Parameter Optimization**: Based on the regime-specific performance, which parameters (thresholds, stop losses, position sizing) appear most problematic?

### Weakness Identification
Please analyze the above data and identify:
- **Single Biggest Weakness**: The most significant performance issue (e.g., "losing money in high-volatility chop", "poor exits in bear markets")
- **Root Cause Hypothesis**: What specific aspect of the strategy logic causes this weakness?
- **Proposed Rule Change**: A concrete, implementable modification to address the weakness

### Market State Analysis
Focus on these specific scenarios:
- High volatility + neutral/choppy markets
- Regime transitions (bull to bear, bear to bull)
- Extended periods in single regimes
- Volatility breakouts and breakdowns

---

*This report contains all necessary data for comprehensive strategy analysis. Copy the relevant sections to your LLM for detailed performance review and weakness identification.*
"""

    # Write the report
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"\n✅ LLM Analysis Report generated: {report_file}")
    print("   Copy relevant sections to your LLM for strategy analysis and weakness identification.")


if __name__ == "__main__":
    sys.exit(main())

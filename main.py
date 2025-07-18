#!/usr/bin/env python3
"""
Trading Data Loader - Command-line tool for fetching and managing market data.

This application provides a flexible system for downloading historical market data
from various sources (currently Alpaca). It supports both initial bulk loading and
incremental updates, with automatic handling of missing data periods.

Features:
- Configuration-driven operation
- Automatic symbol type detection (crypto vs stock)
- Missing bar imputation
- CSV data persistence
- Incremental data updates

Usage:
    python main.py [--config CONFIG_FILE]

Example:
    python main.py --config btc_config.json
"""

import sys
import os
import argparse
from dotenv import load_dotenv
import chalk
from config_manager import Config
from alpaca_data_loader import AlpacaDataLoader

# Load environment variables
load_dotenv()


def main() -> None:
    """
    Main entry point for the trading data loader application.

    Parses command-line arguments, loads configuration, initializes the
    appropriate data loader, and fetches/updates market data. Results are
    displayed to the console with a summary of the operation.
    """
    parser = argparse.ArgumentParser(description="Trading data loader")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="btc_usd_1m.config.json",
        help="Config file name (default: btc_usd_1m.config.json)",
    )
    parser.add_argument(
        "--init", action="store_true", help="Initialize a new configuration file"
    )
    parser.add_argument(
        "--symbol", "-s", type=str, help="Trading symbol (required with --init)"
    )
    parser.add_argument(
        "--timeframe", "-t", type=str, help="Timeframe (required with --init)"
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=10000,
        help="Minimum number of bars to load (default: 10000)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading mode (default: True)",
    )

    args = parser.parse_args()

    # Auto-append .config.json to config argument if not present
    if not args.config.endswith(".config.json"):
        args.config = args.config + ".config.json"

    # Convert hyphens to underscores in config filename
    args.config = args.config.replace("-", "_")

    # Handle --init mode
    if args.init:
        if not args.symbol or not args.timeframe:
            print(
                chalk.red
                + "Error: --symbol and --timeframe are required with --init"
                + chalk.RESET
            )
            parser.print_help()
            sys.exit(1)

        print(
            chalk.green
            + chalk.bold
            + "Trading Data Loader - Config Initialization"
            + chalk.RESET
        )
        print(chalk.blue + f"Symbol: {args.symbol}" + chalk.RESET)
        print(chalk.blue + f"Timeframe: {args.timeframe}" + chalk.RESET)
        print(chalk.blue + f"Min bars: {args.min_bars}" + chalk.RESET)
        print(chalk.blue + f"Paper mode: {args.paper}" + chalk.RESET)

        try:
            config_path = Config.create_config_file(
                args.symbol, args.timeframe, args.min_bars, args.paper
            )
            config_filename = os.path.basename(config_path)
            print(chalk.green + "\n✓ Config file created: {config_path}" + chalk.RESET)
            print(chalk.cyan + "\nTo use this config, run:" + chalk.RESET)
            print(f"  python main.py --config {config_filename}")
        except Exception as e:
            print(chalk.red + f"\n✗ Error creating config: {e}" + chalk.RESET)
            sys.exit(1)
        return

    print(chalk.green + chalk.bold + "Trading Data Loader" + chalk.RESET)

    # Config files are always in the configs/ directory
    config_filename = args.config
    # Remove any path components if user accidentally included them
    config_filename = os.path.basename(config_filename)
    config_path = os.path.join("configs", config_filename)

    print(chalk.blue + f"Config file: {config_filename}" + chalk.RESET)

    # Check if config file exists
    if not os.path.exists(config_path):
        print(
            chalk.red
            + f"\n✗ Error: Config file '{config_filename}' not found in configs/ directory"
            + chalk.RESET
        )
        print(chalk.yellow + "\nAvailable config files:" + chalk.RESET)
        try:
            config_files = [f for f in os.listdir("configs") if f.endswith(".json")]
            if config_files:
                for f in sorted(config_files):
                    print(f"  - {f}")
            else:
                print("  (No config files found)")
        except FileNotFoundError:
            print("  (configs/ directory not found)")
        print(
            chalk.cyan
            + "\nTo create a new config, use: python main.py --init --symbol SYMBOL --timeframe TIMEFRAME"
            + chalk.RESET
        )
        sys.exit(1)

    # Load configuration
    config = Config(config_path)
    print(chalk.cyan + f"Symbol: {config.symbol}" + chalk.RESET)
    print(chalk.cyan + f"Timeframe: {config.timeframe}" + chalk.RESET)
    print(chalk.cyan + f"Min bars: {config.min_bars}" + chalk.RESET)
    print(chalk.cyan + f"Paper mode: {config.paper_mode}" + chalk.RESET)

    # Initialize data loader
    try:
        loader = AlpacaDataLoader(config)
        print(chalk.green + "\n✓ API clients initialized" + chalk.RESET)
        crypto_or_stock = "Crypto" if loader.is_crypto_symbol else "Stock"
        print(chalk.cyan + f"Detected symbol type: {crypto_or_stock}" + chalk.RESET)
    except ValueError as e:
        print(chalk.red + f"\n✗ Error: {e}" + chalk.RESET)
        sys.exit(1)

    # Load data
    print(chalk.yellow + "\nLoading data..." + chalk.RESET)
    try:
        df = loader.load_data(stage_data=True)
        df = loader.clean_data(df, stage_data=True)
        df = loader.transform(df, frac_diff="_fd", log_volume="_lr", stage_data=True)
        print(chalk.green + f"\n✓ Successfully loaded {len(df)} bars" + chalk.RESET)

        # Show summary
        if not df.empty:
            print(chalk.cyan + "\nData Summary:" + chalk.RESET)
            print(f"First timestamp: {df['timestamp'].min()}")
            print(f"Last timestamp: {df['timestamp'].max()}")
            print(f"Total bars: {len(df)}")
            print(f"Missing bars filled: {(df['volume'] == 0).sum()}")

            # Show last few bars
            print(chalk.cyan + "\nLast 5 bars:" + chalk.RESET)
            print(df.tail().to_string(index=False))

            # Process through pipeline if configured
            # DataPipeline.process_from_config(df, config)

    except Exception as e:
        print(chalk.red + f"\n✗ Error loading data: {e}" + chalk.RESET)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

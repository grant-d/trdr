# Trading Data Loader (trdr)

A flexible command-line tool for fetching, managing, and processing historical market data from various sources. Currently supports Alpaca Markets for both stocks and cryptocurrencies.

## Features

- **Multi-source Support**: Built to support multiple data providers (currently Alpaca)
- **Automatic Symbol Detection**: Automatically detects whether a symbol is crypto or stock
- **Incremental Updates**: Smart loading that only fetches new data since last update
- **Missing Data Handling**: Automatically fills missing bars with appropriate values
- **Alternative Bar Types**: Generate dollar bars, volume bars, and tick bars from time-based data
- **Data Pipeline**: Clean and process raw market data with configurable options
- **Configuration-driven**: All settings managed through JSON config files
- **State Tracking**: Persistent state management between runs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trdr.git
cd trdr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Alpaca API keys:
# ALPACA_API_KEY=your_api_key
# ALPACA_SECRET_KEY=your_secret_key
```

## Quick Start

1. Initialize a configuration for Bitcoin 1-minute data:
```bash
python main.py --init --symbol BTC/USD --timeframe 1m
```

2. Fetch the data:
```bash
python main.py --config btc_usd_1m.config.json
```

That's it! Your data will be saved to `data/btc_usd_1m.bars.csv`.

## Usage

### Creating Configurations

Initialize a new configuration file:
```bash
python main.py --init --symbol <SYMBOL> --timeframe <TIMEFRAME> [--min-bars <NUMBER>] [--paper]
```

Parameters:
- `--symbol`: Trading symbol (e.g., "BTC/USD", "ETH/USD", "AAPL", "TSLA")
- `--timeframe`: Time interval - "1m", "5m", "15m", "30m", "1h", "4h", "1d", "3d", "1w"
- `--min-bars`: Minimum number of bars to load (default: 5000)
- `--paper`: Use paper trading mode (default: True)

### Loading Data

Load data using a configuration file:
```bash
python main.py --config <CONFIG_FILE>
```

The config file should be just the filename (e.g., `btc_usd_1m.config.json`). All configs are stored in the `configs/` directory.

### Generating Dollar Bars

Calculate and enable dollar bars for a specific target number:
```bash
python main.py --config btc_usd_1m.config.json --dollar-bars 500
```

This will:
1. Calculate the appropriate dollar threshold for ~500 bars
2. Update your config to enable dollar bar generation
3. Generate the dollar bars on subsequent runs

## Configuration Files

Configuration files are JSON files stored in the `configs/` directory with the following structure:

```json
{
  "symbol": "BTC/USD",
  "timeframe": "1m",
  "min_bars": 5000,
  "paper_mode": true,
  "pipeline": {
    "enabled": false,
    "zero_volume_keep_percentage": 0.1,
    "dollar_bars": {
      "enabled": false,
      "threshold": null,
      "price_column": "close"
    }
  },
  "state": {
    "last_update": "2025-07-15T19:05:00+00:00",
    "total_bars": 1040
  }
}
```

### Pipeline Options

- `enabled`: Enable data cleaning pipeline
- `zero_volume_keep_percentage`: Percentage of zero-volume bars to keep (0.0-1.0)
- `dollar_bars.enabled`: Generate dollar bars
- `dollar_bars.threshold`: Dollar volume threshold for bar generation (calculated by --dollar-bars or set manually)
- `dollar_bars.price_column`: Price column for calculations ("close", "hlc3", "ohlc4", etc.)

## Data Processing Pipeline

The data pipeline provides several cleaning and transformation options:

1. **Missing Value Handling**: Forward-fills price data, fills volume with 0
2. **Outlier Detection**: Clamps extreme values using rolling statistics
3. **Price Validation**: Ensures logical price relationships (high >= low, etc.)
4. **Zero Volume Filtering**: Removes excessive zero-volume bars while keeping some for continuity
5. **Alternative Bar Generation**: Creates dollar bars, volume bars, or tick bars

### Enabling the Pipeline

Edit your config file or use the `--dollar-bars` option to enable pipeline processing:

```json
{
  "pipeline": {
    "enabled": true,
    "zero_volume_keep_percentage": 0.1,
    "dollar_bars": {
      "enabled": true,
      "threshold": 750.25,
      "price_column": "hlc3"
    }
  }
}
```

## Output Files

Files are organized with standardized naming using dot separators:

### Data Files (`data/` directory)
- **Time bars**: `{symbol}_{timeframe}.bars.csv` (e.g., `btc_usd_1m.bars.csv`)
- **Cleaned data**: `{symbol}_{timeframe}.cleaned.csv`
- **Dollar bars**: `{symbol}_{timeframe}.dollar-{threshold}.csv` (e.g., `btc_usd_1m.dollar-443k.csv`)
- **Volume bars**: `{symbol}_{timeframe}.volume-{threshold}.csv`

### Configuration Files (`configs/` directory)
- **Configuration**: `{symbol}_{timeframe}.config.json` (e.g., `btc_usd_1m.config.json`)
- **Runtime State**: `{symbol}_{timeframe}.state.json` (e.g., `btc_usd_1m.state.json`)

The state file contains optimization history, hall of fame, and other runtime data separate from configuration.

## Data Format

All CSV files contain the following columns:
- `timestamp`: Bar timestamp (timezone-aware UTC)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `trade_count`: Number of trades
- `vwap`: Volume-weighted average price (if available)

Additional calculated columns in processed data:
- `hlc3`: (high + low + close) / 3
- `dv`: Dollar volume (price × volume)

## API Requirements

### Alpaca Markets
- API Key and Secret Key required
- Supports both live and paper trading endpoints
- Free tier available with rate limits
- Covers US stocks and major cryptocurrencies

## Project Structure

```
trdr/
├── main.py                 # Main entry point
├── config_manager.py       # Configuration management with Pydantic
├── state.py               # State tracking and persistence
├── alpaca_data_loader.py  # Alpaca API integration
├── data_pipeline.py       # Data cleaning and transformation
├── bar_aggregators.py     # Alternative bar generation
├── filename_utils.py      # File naming utilities
├── configs/               # Configuration files
│   └── *.json
├── data/                  # Downloaded and processed data
│   └── *.csv
└── requirements.txt       # Python dependencies
```

## Advanced Usage

### Custom Bar Aggregators

The system supports three types of alternative bars:

1. **Dollar Bars**: Aggregate by dollar volume (price × volume)
2. **Volume Bars**: Aggregate by volume only
3. **Tick Bars**: Aggregate by number of trades

Each aggregator can estimate appropriate thresholds:

```python
from bar_aggregators import DollarBarAggregator

# Estimate threshold for ~1000 dollar bars
threshold = DollarBarAggregator.estimate_threshold(df, target_bars=1000)
```

### Extending Data Sources

To add a new data source:

1. Create a new loader class inheriting from a base loader interface
2. Implement required methods: `load_data()`, `get_historical_bars()`, etc.
3. Handle symbol type detection and data formatting
4. Update main.py to support the new source

## Troubleshooting

### Common Issues

1. **API Authentication Errors**
   - Verify your API keys in `.env`
   - Check if using correct endpoint (paper vs live)

2. **Missing Data**
   - Some symbols may have limited historical data
   - Crypto data typically available 24/7
   - Stock data only available during market hours

3. **Rate Limiting**
   - Alpaca free tier has rate limits
   - The loader automatically handles rate limiting with retries

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# trdr - Trading Data Pipeline

A desktop CLI application for processing trading data that handles both historical backtesting and live data feeds.

## Overview

`trdr` is a command-line tool that:
- Reads market data from CSV/Parquet files or exchanges (Coinbase, etc.)
- Transforms data through a configurable pipeline
- Outputs to CSV/Parquet/SQLite for analysis
- Runs as either a one-shot CLI tool or continuous local process

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trdr.git
cd trdr

# Install dependencies
yarn install

# Build the project
yarn build
```

## Usage

```bash
# Process a pipeline configuration
trdr --config configs/aggregate-5m.json

# Override configuration values
trdr --config configs/collect-coinbase.json --override symbols=BTC-USD,ETH-USD

# Run in continuous mode
trdr --config configs/live-pipeline.json --mode continuous
```

## Development

```bash
# Run in development mode
yarn dev

# Run tests
yarn test

# Check types
yarn typecheck

# Lint code
yarn lint
```

## Project Structure

```
src/
├── commands/        # CLI command implementations
├── config/          # Configuration handling
├── interfaces/      # TypeScript interfaces
├── models/          # Data models and DTOs
├── providers/       # Data provider implementations
├── repositories/    # Storage layer implementations
├── transforms/      # Data transformation implementations
└── utils/           # Utility functions and helpers
```

## Configuration

Pipelines are configured using JSON files. See the `configs/` directory for examples.

### Example Configuration

```json
{
  "name": "5-minute aggregation",
  "input": {
    "type": "file",
    "format": "csv",
    "path": "data/1m-bars.csv"
  },
  "transformations": [{
    "type": "timeframeAggregation",
    "params": { "targetTimeframe": "5m" },
    "enabled": true
  }],
  "output": {
    "type": "csv",
    "path": "data/5m-bars.csv"
  }
}
```

## Environment Variables

For exchange connections, set the appropriate API keys:

```bash
export COINBASE_API_KEY=your_api_key
export COINBASE_API_SECRET=your_api_secret
```

## Features

### Data Providers
- **File Provider**: Read from CSV and Parquet files
- **Coinbase Provider**: Real-time and historical data via REST/WebSocket

### Transformations
- **Data Cleaning**: Handle missing values, outliers
- **Aggregation**: Convert to any timeframe (1m → 5m, 17m, etc.)
- **Normalization**: Log returns, z-score, min-max scaling
- **Technical Indicators**: Moving averages, RSI, Bollinger Bands, MACD
- **Alternative Bars**: Tick bars, volume bars, dollar bars

### Storage Options
- **CSV**: Simple, human-readable output
- **Parquet**: Efficient columnar storage
- **SQLite**: Indexed database for queries

## License

ISC
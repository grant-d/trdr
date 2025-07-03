# TRDR CLI

Command-line interface for running the TRDR trading system.

## Quick Start

### 1. Install Dependencies

From the repository root:

```bash
yarn install
```

### 2. Build the Project

```bash
yarn build
```

### 3. Run Backtest with CSV Data

From the CLI package directory (`packages/cli`):

```bash
# Using the provided sample data
yarn dev backtest -f sample-data.csv

# Using your own CSV file
yarn dev backtest -f /path/to/your/data.csv -s BTC-USD -c 10000

# With specific date range
yarn dev backtest -f data.csv --start-date 2024-01-01 --end-date 2024-01-31

# Verbose output
yarn dev backtest -f data.csv -v
```

## CSV Format

Your CSV file should have the following columns:

```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42000,42500,41800,42300,1500
```

Supported timestamp formats:
- ISO 8601: `2024-01-01T00:00:00Z`
- Separate date/time columns: `date,time,open,high,low,close,volume`
- Unix timestamp: `1704067200000,open,high,low,close,volume`

## Command Options

### Backtest

```bash
trdr backtest [options]
```

Options:
- `-f, --file <path>` - Path to CSV file (default: "./data.csv")
- `-s, --symbol <symbol>` - Trading symbol (default: "BTC-USD")
- `-c, --capital <amount>` - Initial capital (default: "10000")
- `--grid-spacing <percent>` - Grid spacing percentage (default: "2")
- `--grid-levels <count>` - Number of grid levels (default: "10")
- `--agents <list>` - Comma-separated list of agents (default: "rsi,macd,bollinger")
- `--start-date <date>` - Start date in YYYY-MM-DD format
- `--end-date <date>` - End date in YYYY-MM-DD format
- `-v, --verbose` - Show detailed output

## Available Agents

- `rsi` - RSI (Relative Strength Index) based signals
- `macd` - MACD (Moving Average Convergence Divergence) signals
- `bollinger` - Bollinger Bands mean reversion signals

## Example Output

```
✔ Backtest environment ready

Running backtest...
✔ Backtest completed

=== Backtest Results ===

┌─────────────────┬──────────────┐
│ Metric          │ Value        │
├─────────────────┼──────────────┤
│ Initial Capital │ $10000.00    │
│ Final Capital   │ $10523.45    │
│ Total P&L       │ +$523.45     │
│ ROI             │ +5.23%       │
│ Total Trades    │ 42           │
│ Winning Trades  │ 28           │
│ Win Rate        │ 66.67%       │
│ Avg Trade P&L   │ +$12.46      │
│ Duration        │ 4.2s         │
│ Candles/sec     │ 2380         │
└─────────────────┴──────────────┘
```

## Development

To run in development mode with hot reloading:

```bash
yarn dev backtest -f sample-data.csv -v
```

## Notes

- The backtest runs as fast as possible (no replay delay)
- Grid trading is automatically configured based on your parameters
- Multiple agents work together to generate consensus signals
- Position sizing uses Kelly Criterion with risk management
- All trades are simulated - no real money is used in backtest mode
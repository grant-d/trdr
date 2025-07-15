# Live Paper Trading System

## Overview

The live trader is a comprehensive paper trading system that runs genetic algorithm (GA) optimization on every new bar arrival. It features CSV caching for historical data, automatic gap filling, and configurable parameters for both trading and optimization.

## Features

- **Real-time GA Optimization**: Runs optimization on every new bar to adapt to changing market conditions
- **CSV Data Caching**: Stores historical data locally for efficiency and fast restarts
- **Automatic Gap Filling**: Detects and fills missing data when restarting
- **Paper Trading Only**: Simulates trades without placing real orders
- **Sound Alerts**: Audio notifications for buy/sell signals
- **Configurable Parameters**: Full control over GA settings, capital, and timing

## Usage

### Basic Usage

```bash
# Simple run with defaults
./run_live.sh BTCUSD

# With custom timeframe
./run_live.sh MSFT 5
```

### Advanced Usage with Named Parameters

```bash
# Using run_live_custom.sh for better control
./run_live_custom.sh BTCUSD --population 50 --generations 20 --timeframe 5

# Full example with all parameters
./run_live_custom.sh ETH/USD \
    --timeframe 15 \
    --population 100 \
    --generations 50 \
    --lookback 500 \
    --initial-bars 4000 \
    --capital 50000 \
    --check-interval 10
```

### Command Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timeframe` | 1 | Bar timeframe in minutes |
| `--population` | 50 | GA population size |
| `--generations` | 20 | Number of GA generations |
| `--lookback` | 200 | Bars to use for optimization |
| `--initial-bars` | 1000 | Historical bars to fetch on first run |
| `--capital` | 100000 | Initial capital for simulation |
| `--check-interval` | 5 | Seconds between checking for new bars |

## CSV Caching

### Cache Location
All cache files are stored in: `./data/cache/`

### File Naming Convention
- Stock: `MSFT_1min.csv`
- Crypto: `BTC_USD_5min.csv`

### Cache Behavior
1. **First Run**: Downloads `initial-bars` worth of historical data
2. **Subsequent Runs**: Loads from cache and fills any gaps
3. **Live Updates**: Appends new bars as they arrive

## Trading Logic

### GA Optimization
- Runs on every new bar arrival
- Optimizes parameters:
  - Entry/exit Z-scores
  - Stop loss percentages
  - Position sizing
  - Trailing stop settings

### Signal Generation
- Uses mean reversion strategy with Z-score thresholds
- Dynamic parameter adjustment based on recent market behavior
- Risk management with stop loss and trailing stops

## Examples

### Quick Test (1-minute bars)
```bash
./run_live.sh BTCUSD 1
```

### Production Settings (5-minute bars, larger GA)
```bash
./run_live_custom.sh BTCUSD \
    --timeframe 5 \
    --population 50 \
    --generations 30 \
    --lookback 300
```

### Conservative Trading (less capital, tighter stops)
```bash
./run_live_custom.sh MSFT \
    --capital 25000 \
    --population 30 \
    --generations 20
```

## Performance Considerations

- **Population Size**: Larger = better optimization but slower
- **Generations**: More = better convergence but slower
- **Check Interval**: Lower = more responsive but higher CPU usage
- **Initial Bars**: More = better initial analysis but longer startup

## Recommended Settings

### Fast Testing
- Population: 20
- Generations: 10
- Timeframe: 1 minute
- Check interval: 5 seconds

### Production
- Population: 50-100
- Generations: 30-50
- Timeframe: 5-15 minutes
- Check interval: 10-30 seconds

## Troubleshooting

### No data appearing
- Check API keys in `.env` file
- Verify symbol format (crypto needs slash: BTC/USD)
- Check internet connection

### Slow optimization
- Reduce population size
- Reduce generations
- Increase timeframe

### Missing historical data
- Delete cache file to force fresh download
- Increase `initial-bars` parameter
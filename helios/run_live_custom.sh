#!/bin/bash

# Custom Live Paper Trading Script with named parameters
# Example usage:
#   ./run_live_custom.sh BTCUSD --population 50 --generations 20 --timeframe 5
#   ./run_live_custom.sh MSFT --capital 50000 --lookback 500

# Activate virtual environment
if [ -f ../.venv/bin/activate ]; then
    source ../.venv/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if required environment variables are set
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file"
    exit 1
fi

# Get symbol (required first argument)
if [ -z "$1" ]; then
    echo "Usage: $0 SYMBOL [options]"
    echo ""
    echo "Options:"
    echo "  --timeframe MINUTES      Bar timeframe in minutes (default: 1)"
    echo "  --population SIZE        GA population size (default: 20)"
    echo "  --generations COUNT      GA generations (default: 10)"
    echo "  --lookback BARS         Lookback bars for optimization (default: 200)"
    echo "  --initial-bars COUNT    Initial historical bars to fetch (default: 4000)"
    echo "  --capital AMOUNT        Initial capital (default: 100000)"
    echo "  --check-interval SECS   Check interval in seconds (default: 5)"
    echo "  --max-opt-bars BARS     Max bars for optimization window (default: 2000)"
    echo ""
    echo "Examples:"
    echo "  $0 BTCUSD --population 50 --generations 20"
    echo "  $0 MSFT --timeframe 5 --capital 50000"
    exit 1
fi

SYMBOL=$1
shift  # Remove symbol from arguments

# Default values
TIMEFRAME=1
POPULATION=20
GENERATIONS=10
LOOKBACK=200
INITIAL_BARS=1000
CAPITAL=100000
CHECK_INTERVAL=5
MAX_OPT_BARS=2000

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeframe)
            TIMEFRAME="$2"
            shift 2
            ;;
        --population)
            POPULATION="$2"
            shift 2
            ;;
        --generations)
            GENERATIONS="$2"
            shift 2
            ;;
        --lookback)
            LOOKBACK="$2"
            shift 2
            ;;
        --initial-bars)
            INITIAL_BARS="$2"
            shift 2
            ;;
        --capital)
            CAPITAL="$2"
            shift 2
            ;;
        --check-interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        --max-opt-bars)
            MAX_OPT_BARS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert crypto symbols to correct format (BTCUSD -> BTC/USD)
if [[ $SYMBOL == *"USD" ]] && [[ $SYMBOL != *"/"* ]]; then
    # Extract base currency and add slash
    BASE="${SYMBOL%USD}"
    SYMBOL="${BASE}/USD"
    echo "Converted symbol to: $SYMBOL"
fi

# Determine if crypto
if [[ $SYMBOL == *"USD"* ]] || [[ $SYMBOL == *"/"* ]]; then
    CRYPTO_FLAG="--crypto"
else
    CRYPTO_FLAG=""
fi

echo "Starting LIVE paper trader with CSV caching..."
echo "Symbol: $SYMBOL"
echo "Mode: $([ -n "$CRYPTO_FLAG" ] && echo "Crypto" || echo "Stock")"
echo ""
echo "Configuration:"
echo "  Timeframe: ${TIMEFRAME} minutes"
echo "  Population: $POPULATION"
echo "  Generations: $GENERATIONS"
echo "  Lookback: $LOOKBACK bars"
echo "  Initial bars: $INITIAL_BARS"
echo "  Capital: \$$CAPITAL"
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Max optimization bars: $MAX_OPT_BARS"
echo ""
echo "Features:"
echo "  - GA optimization on every new bar"
echo "  - CSV caching for efficiency"
echo "  - Automatic gap filling on restart"
echo "  - Sound alerts for trades"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python live_trader.py "$SYMBOL" \
    --api-key "$ALPACA_API_KEY" \
    --secret-key "$ALPACA_SECRET_KEY" \
    --timeframe "$TIMEFRAME" \
    --population "$POPULATION" \
    --generations "$GENERATIONS" \
    --lookback "$LOOKBACK" \
    --initial-bars "$INITIAL_BARS" \
    --capital "$CAPITAL" \
    --check-interval "$CHECK_INTERVAL" \
    --max-opt-bars "$MAX_OPT_BARS" \
    $CRYPTO_FLAG
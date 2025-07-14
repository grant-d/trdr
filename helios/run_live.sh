#!/bin/bash

# Live Paper Trading Script with CSV caching
# Uses GA optimization on every bar with data caching
#
# Usage: ./run_live.sh SYMBOL [TIMEFRAME] [POPULATION] [GENERATIONS] [LOOKBACK] [INITIAL_BARS] [CAPITAL] [CHECK_INTERVAL] [MAX_OPT_BARS]
#
# Example: ./run_live.sh BTCUSD 5 50 20 300 2000 50000 10 2000
#
# For named parameters, use run_live_custom.sh instead

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

# Default values
SYMBOL=${1:-"MSFT"}
TIMEFRAME=${2:-1}  # Default to 1-minute bars

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

echo "Starting ADAPTIVE live paper trader with CSV caching..."
echo "Symbol: $SYMBOL"
echo "Bar timeframe: $TIMEFRAME minutes"
echo "Mode: $([ -n "$CRYPTO_FLAG" ] && echo "Crypto" || echo "Stock")"
echo ""
echo "Features:"
echo "  - GA optimization on every new bar"
echo "  - CSV caching for efficiency"
echo "  - Automatic gap filling on restart"
echo "  - Sound alerts for trades"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Additional optional parameters
POPULATION=${3:-50}
GENERATIONS=${4:-20}
LOOKBACK=${5:-200}
INITIAL_BARS=${6:-1000}
CAPITAL=${7:-100000}
CHECK_INTERVAL=${8:-5}
MAX_OPT_BARS=${9:-2000}

# Display parameters if customized
if [ "$#" -gt 2 ]; then
    echo "Custom parameters:"
    echo "  Population: $POPULATION"
    echo "  Generations: $GENERATIONS"
    echo "  Lookback: $LOOKBACK bars"
    echo "  Initial bars: $INITIAL_BARS"
    echo "  Capital: \$$CAPITAL"
    echo "  Check interval: ${CHECK_INTERVAL}s"
    echo "  Max optimization bars: $MAX_OPT_BARS"
    echo ""
fi

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
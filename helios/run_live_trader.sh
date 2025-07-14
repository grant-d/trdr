#!/bin/bash

# Live Paper Trading Script
# Uses GA optimization for real-time trading decisions

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
INTERVAL=${2:-60}
TIMEFRAME=${3:-1}  # Default to 1-minute bars

# Determine if crypto
if [[ $SYMBOL == *"USD"* ]] || [[ $SYMBOL == *"/"* ]]; then
    CRYPTO_FLAG="--crypto"
else
    CRYPTO_FLAG=""
fi

echo "Starting live paper trader..."
echo "Symbol: $SYMBOL"
echo "Update interval: $INTERVAL seconds"
echo "Bar timeframe: $TIMEFRAME minutes"
echo "Mode: $([ -n "$CRYPTO_FLAG" ] && echo "Crypto" || echo "Stock")"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Check if adaptive mode is requested
if [ "$4" == "adaptive" ]; then
    echo "Running ADAPTIVE trader with GA optimization..."
    python live_trader_adaptive.py "$SYMBOL" \
        --api-key "$ALPACA_API_KEY" \
        --secret-key "$ALPACA_SECRET_KEY" \
        --timeframe "$TIMEFRAME" \
        $CRYPTO_FLAG
else
    python live_trader_simple.py "$SYMBOL" \
        --api-key "$ALPACA_API_KEY" \
        --secret-key "$ALPACA_SECRET_KEY" \
        --interval "$INTERVAL" \
        --timeframe "$TIMEFRAME" \
        $CRYPTO_FLAG
fi
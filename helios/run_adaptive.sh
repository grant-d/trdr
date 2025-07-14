#!/bin/bash

# Adaptive Live Paper Trading Script with CSV caching
# Uses GA optimization on every bar with data caching

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

python live_trader_adaptive.py "$SYMBOL" \
    --api-key "$ALPACA_API_KEY" \
    --secret-key "$ALPACA_SECRET_KEY" \
    --timeframe "$TIMEFRAME" \
    $CRYPTO_FLAG
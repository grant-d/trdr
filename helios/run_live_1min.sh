#!/bin/bash

# Live paper trading with 1-minute bars for faster testing
# Usage: ./run_live_1min.sh [SYMBOL] [UPDATE_INTERVAL]

SYMBOL=${1:-"MSFT"}
INTERVAL=${2:-30}  # Update every 30 seconds for 1-minute bars

echo "=== FAST TESTING MODE ==="
echo "Using 1-minute bars for quicker signal generation"
echo ""

./run_live_trader.sh "$SYMBOL" "$INTERVAL" 1
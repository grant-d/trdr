#!/bin/bash

# Example: Live paper trading for BTCUSD (Bitcoin)

echo "Starting live paper trading for BTCUSD..."
echo "This will trade crypto using default strategy parameters."
echo "Press Ctrl+C to stop at any time."
echo ""

# For live trader with CSV caching
echo "Note: For live trader with CSV caching, use:"
echo "  ./run_live.sh BTCUSD 1"
echo ""

# Run with 60-second update interval for crypto
./run_live_trader.sh BTCUSD 60
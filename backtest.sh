#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
else
    source .venv/bin/activate
fi

# Check for .env
if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    echo "Copy .env.example to .env and add your Alpaca API keys"
    exit 1
fi

# Default values
SYMBOL="${SYMBOL:-crypto:BTC/USD}"
LOOKBACK="${LOOKBACK:-500}"
FOLDS="${FOLDS:-0}"

# Show help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: ./backtest.sh [OPTIONS]"
    echo ""
    echo "Environment variables:"
    echo "  SYMBOL    Asset symbol (default: crypto:BTC/USD)"
    echo "  LOOKBACK  Number of bars (default: 500)"
    echo "  FOLDS     Walk-forward folds, 0=single (default: 0)"
    echo ""
    echo "Examples:"
    echo "  ./backtest.sh -v"
    echo "  SYMBOL=AAPL LOOKBACK=1000 ./backtest.sh"
    echo "  FOLDS=5 ./backtest.sh --output results.json"
    echo ""
    echo "All other options passed to python -m trdr.backtest"
    exit 0
fi

exec python -m trdr.backtest \
    --symbol "$SYMBOL" \
    --lookback "$LOOKBACK" \
    --folds "$FOLDS" \
    "$@"

#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Show help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: ./run.sh"
    echo ""
    echo "Runs the trading bot TUI with live/paper trading."
    echo "Configure via .env file (see .env.example)."
    exit 0
fi

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

exec python -m trdr.main "$@"

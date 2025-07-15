#!/bin/bash

# Trading Data Loader Runner Script
# Activates virtual environment and runs main.py with optional config

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo -e "${GREEN}Trading Data Loader Runner${NC}"
    echo "This script activates the virtual environment and runs main.py"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options are passed directly to main.py:"
    # Try to get help from main.py without venv (just show the message)
    echo ""
    source "$SCRIPT_DIR/.venv/bin/activate" 2>/dev/null && python "$SCRIPT_DIR/main.py" -h 2>/dev/null && deactivate 2>/dev/null || {
        echo "  -h, --help            show this help message and exit"
        echo "  --config CONFIG, -c CONFIG"
        echo "                        Path to config file (default: btc_usd_1m_config.json)"
        echo "  --init                Initialize a new configuration file"
        echo "  --symbol SYMBOL, -s SYMBOL"
        echo "                        Trading symbol (required with --init)"
        echo "  --timeframe TIMEFRAME, -t TIMEFRAME"
        echo "                        Timeframe (required with --init)"
        echo "  --min-bars MIN_BARS   Minimum number of bars to load (default: 1000)"
        echo "  --paper               Use paper trading mode (default: True)"
    }
    exit 0
fi

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at $SCRIPT_DIR/.venv${NC}"
    echo -e "${YELLOW}Please create a virtual environment first:${NC}"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$SCRIPT_DIR/.venv/bin/activate"

# Check if activation was successful
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}Virtual environment activated: $VIRTUAL_ENV${NC}"

# Run main.py with all passed arguments
echo -e "${BLUE}Running trading data loader...${NC}"
python "$SCRIPT_DIR/main.py" "$@"

# Capture exit code
EXIT_CODE=$?

# Deactivate virtual environment
deactivate 2>/dev/null || true

# Exit with same code as main.py
exit $EXIT_CODE
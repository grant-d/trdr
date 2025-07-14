#!/bin/bash

# Script to run optimization and test for both BTC and MSFT

echo "========================================"
echo "HELIOS OPTIMIZATION AND TEST"
echo "========================================"

# Try to find a working Python
PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "Error: No Python found"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""

# Change to helios directory
cd /Users/grantdickinson/repos/trdr/helios

# MSFT Optimization
echo "========================================"
echo "OPTIMIZING MSFT"
echo "========================================"
echo "Running: $PYTHON main.py optimize --data ../data/MSFT.csv --population 20 --generations 10 --output msft --allow-shorts"
$PYTHON main.py optimize --data ../data/MSFT.csv --population 20 --generations 10 --output msft --allow-shorts

echo ""
echo "========================================"
echo "TESTING MSFT"
echo "========================================"
if [ -f "optimization_results/msft-params.json" ]; then
    echo "Running: $PYTHON main.py test --params optimization_results/msft-params.json"
    $PYTHON main.py test --params optimization_results/msft-params.json
    
    # Check thresholds
    echo ""
    echo "MSFT Threshold Check:"
    cat optimization_results/msft-params.json | grep -A 6 "bull_threshold" | grep -E "(strong_bull|weak_bull|neutral|bear)" || echo "Could not extract thresholds"
else
    echo "ERROR: MSFT optimization failed - no params file found"
fi

# BTC Optimization
echo ""
echo "========================================"
echo "OPTIMIZING BTC"
echo "========================================"
echo "Running: $PYTHON main.py optimize --data ../data/BTCUSD.csv --population 20 --generations 10 --output btc --allow-shorts"
$PYTHON main.py optimize --data ../data/BTCUSD.csv --population 20 --generations 10 --output btc --allow-shorts

echo ""
echo "========================================"
echo "TESTING BTC"
echo "========================================"
if [ -f "optimization_results/btc-params.json" ]; then
    echo "Running: $PYTHON main.py test --params optimization_results/btc-params.json"
    $PYTHON main.py test --params optimization_results/btc-params.json
    
    # Check thresholds
    echo ""
    echo "BTC Threshold Check:"
    cat optimization_results/btc-params.json | grep -A 6 "bull_threshold" | grep -E "(strong_bull|weak_bull|neutral|bear)" || echo "Could not extract thresholds"
else
    echo "ERROR: BTC optimization failed - no params file found"
fi

echo ""
echo "========================================"
echo "COMPLETE"
echo "========================================"
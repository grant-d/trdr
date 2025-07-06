#!/bin/bash

# Run the BTC Alpaca pipeline
# Make sure ALPACA_API_KEY and ALPACA_API_SECRET are set in your environment

echo "Running BTC Lorentzian Distance Analysis Pipeline..."
echo "Fetching 1 year of daily BTC data from Alpaca..."

# Create output directory if it doesn't exist
mkdir -p ./output

# Run the pipeline
yarn cli -c ./configs/btc-alpaca-pipeline.json

echo "Pipeline complete! Check ./output/btc-lorenz-1y.csv for results."
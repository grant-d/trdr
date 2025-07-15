#!/usr/bin/env python3
"""
Main entry point for the trading application
"""

import os
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import chalk

# Load environment variables
load_dotenv()


def main():
    """Main function"""
    print(chalk.green + chalk.bold + "Trading Application Started" + chalk.RESET)
    print(chalk.blue + f"Python {sys.version.split()[0]}" + chalk.RESET)
    print(chalk.blue + f"Working Directory: {os.getcwd()}" + chalk.RESET)
    
    # Example: Check if API keys are loaded
    if os.getenv("ALPACA_API_KEY"):
        print(chalk.green + "✓ Alpaca API key loaded" + chalk.RESET)
    else:
        print(chalk.yellow + "⚠ Alpaca API key not found in .env" + chalk.RESET)
    
    # Example: Create a simple DataFrame
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'price': np.random.uniform(100, 200, 3),
        'volume': np.random.randint(1000000, 10000000, 3)
    })
    
    print(chalk.cyan + "\nSample Market Data:" + chalk.RESET)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
# AAPL HMA Bull Put Spread Backtest

A backtesting system for trading Bull Put Spreads on AAPL using Hull Moving Average (HMA) trend signals.

## Strategy Overview

**Entry Conditions:**
- HMA 50 is trending up
- HMA 200 is trending up

**Position:**
- Sell a put option ~30 delta (just below current price)
- Buy a put option $5 below the short put (for protection)
- This creates a Bull Put Spread (credit spread)

**Exit Conditions:**
- Profit target: 50% of maximum profit
- Stop loss: 2x the credit received
- Trend reversal: Either HMA turns down
- Expiration: 30 days to expiration (DTE)

## Project Structure

```
aapl_hma_backtest/
├── data/                   # Data loading and management
│   ├── __init__.py
│   └── data_loader.py     # Fetch historical data (yfinance)
├── indicators/             # Technical indicators
│   ├── __init__.py
│   └── hma.py            # Hull Moving Average calculation
├── options/               # Options pricing and positions
│   ├── __init__.py
│   ├── black_scholes.py  # Black-Scholes pricing model
│   └── position.py       # Position and portfolio management
├── strategy/              # Trading strategy
│   ├── __init__.py
│   └── hma_bull_put_strategy.py  # Main strategy logic
├── reports/               # Output files (CSV, charts)
├── run_backtest.py       # Main script to run backtest
└── requirements.txt      # Python dependencies
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. **IMPORTANT: Set up Alpaca for REAL data access**

To fetch REAL AAPL historical data, you need Alpaca API credentials with data subscription enabled:

### Getting Alpaca Credentials:
1. Go to [https://alpaca.markets/](https://alpaca.markets/) and sign up (FREE)
2. Navigate to your dashboard
3. Go to **"Market Data" → "Subscriptions"**
4. **Enable the IEX data feed** (it's FREE - no credit card required)
5. Wait 2-3 minutes for the subscription to activate
6. Get your API keys from "Your API Keys" section

### Set Environment Variables:
```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_API_SECRET="your_api_secret_here"
```

Or create a `.env` file (see `.env.example`)

**NOTE:** Without enabling the IEX data subscription, you'll get "Access denied" errors and the backtest will use synthetic data instead of real AAPL prices.

## Usage

Run the backtest:
```bash
python run_backtest.py
```

This will:
1. Download REAL historical AAPL data from Alpaca (2020-2024)
2. Calculate HMA 50 and HMA 200
3. Simulate bull put spread trades
4. Generate performance statistics
5. Create visualizations
6. Save results to CSV files

## Strategy Parameters

- **Initial Capital:** $100,000
- **Short Put Delta:** -0.30 (30 delta)
- **Spread Width:** $5.00
- **Days to Expiration:** 30 days
- **Profit Target:** 50% of max profit
- **Stop Loss:** 2x credit received
- **Max Positions:** 5 concurrent positions
- **Risk-Free Rate:** 5%

## Output

The backtest generates:
- Trade log CSV file
- Equity curve CSV file
- Performance charts (price, equity, P&L)
- Statistics summary in console

## Hull Moving Average (HMA)

The Hull Moving Average is designed to reduce lag while maintaining smoothness:

**Formula:** HMA(n) = WMA(2 × WMA(n/2) - WMA(n), √n)

Where WMA is the Weighted Moving Average.

## Bull Put Spread

A bull put spread is a credit spread strategy:
- **Sell** higher strike put (collect premium)
- **Buy** lower strike put (limit risk)

**Maximum Profit:** Credit received
**Maximum Loss:** Spread width - credit received
**Breakeven:** Short strike - credit received

## Risk Disclaimer

This is for educational and backtesting purposes only. Past performance does not guarantee future results. Options trading involves substantial risk and is not suitable for all investors.

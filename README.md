# trdr

Trading algorithms research and development framework.

## Setup

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and add your Alpaca credentials:

```bash
cp .env.example .env
```

1. Edit `.env` with your API keys:

```bash
# Required - get from https://app.alpaca.markets/paper/dashboard/overview
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# Optional - defaults to paper trading
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional - trading symbol (default: AAPL)
# Crypto: crypto:BTC/USD, crypto:ETH/USD
# Stock: stock:AAPL or just AAPL
SYMBOL=crypto:BTC/USD

# Optional - backtest timeframe (default: 1h)
# Examples: 1m, 5m, 15m, 30m, 1h, 4h, 1d
BACKTEST_TIMEFRAME=1h
```

## Running Tests

Run backtest with default settings (BTC/USD, 1h bars):

```bash
.venv/bin/python -m pytest tests/test_volume_area_breakout.py -v
```

Override symbol and timeframe:

```bash
BACKTEST_SYMBOL=stock:AAPL BACKTEST_TIMEFRAME=4h .venv/bin/python -m pytest tests/test_volume_area_breakout.py -v
```

Timeframe formats: `1h`, `4h`, `15m`, `1d`, `hour`, `minute`, `day`

## SICA Optimization

Run SICA loop to optimize the Volume Area Breakout strategy:

```bash
/sica:sica-loop aapl-1d      # Start loop with config
/sica:sica-continue aapl-1d  # Add more iterations to completed run
/sica:sica-status            # Check current progress
```

See `.sica/configs/` for available configs and `plugins/sica/README.md` for full documentation.

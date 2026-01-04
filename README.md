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
```

## Running Tests

Run strategy benchmark (symbol/timeframe defined in sica_bench.py):

```bash
.venv/bin/python -m pytest tests/test_volume_area_breakout.py -v
```

Override symbol (timeframe is code-driven in sica_bench.py):

```bash
BACKTEST_SYMBOL=stock:AAPL .venv/bin/python src/trdr/strategy/volume_area_breakout/sica_bench.py
```

Timeframe can be overridden via env var if needed:

```bash
BACKTEST_SYMBOL=crypto:ETH/USD BACKTEST_TIMEFRAME=4h .venv/bin/python src/trdr/strategy/volume_area_breakout/sica_bench.py
```

Timeframe formats: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`

## SICA Optimization

Run SICA loop to optimize the Volume Area Breakout strategy:

```bash
/sica:sica-loop aapl-1d      # Start loop with config
/sica:sica-continue aapl-1d  # Add more iterations to completed run
/sica:sica-status            # Check current progress
```

See `.sica/configs/` for available configs and `plugins/sica/README.md` for full documentation.

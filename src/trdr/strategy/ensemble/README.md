# Ensemble Random Forest Strategy

Machine learning strategy using random forest trained on multiple technical indicators.

## Features

Combines 5 technical indicators into a weighted ensemble:

1. **Laguerre RSI**: Adaptive RSI with reduced lag
2. **RVI**: Relative Vigor Index (momentum oscillator)
3. **POC**: Point of Control from volume profile (price distance)
4. **Squeeze**: Squeeze Momentum Indicator (volatility/momentum)
5. **Volume Trend**: Accumulation/distribution signals

## How It Works

1. **Feature Extraction**: Real-time calculation of all 5 indicators
2. **Training**: Random forest learns patterns from historical bars
3. **Prediction**: Model predicts profitable entry points (confidence threshold: 60%)
4. **Exits**: Model signals when to exit positions

## Configuration

```python
from trdr.strategy.ensemble import EnsembleConfig, EnsembleStrategy

config = EnsembleConfig(
    symbol="crypto:BTC/USD",
    timeframe="15m",
    lookback_bars=100,      # Bars for training
    retrain_every=0,        # 0 = train once, >0 = retrain interval
    n_estimators=50,        # Number of trees
    max_tree_depth=5,       # Max tree depth
    min_samples_split=20,   # Min samples per split
)

strategy = EnsembleStrategy(config)
```

## Model Persistence

- Trained model saved to `ensemble_model.pkl`
- Automatically loads on restart (no retraining needed)
- Set `retrain_every > 0` for periodic retraining

## Advantages

- **Robust**: Random forest handles noisy financial data better than single tree
- **Adaptive**: Learns from historical patterns
- **Multi-signal**: Combines multiple indicators intelligently
- **Interpretable**: Feature importances show which indicators matter most

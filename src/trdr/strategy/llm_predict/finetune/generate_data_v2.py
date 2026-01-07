"""Generate training data V2 - Rich outputs + engineered features.

Key insight from TimeSense paper: binary classification provides too thin
a gradient for learning temporal patterns. Need richer supervision.

Changes:
1. INPUT: Add engineered features (trend, volatility, momentum)
2. OUTPUT: Predict multiple targets (change%, next_price, pattern_description)

This forces the model to actually learn patterns, not just memorize label distributions.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
DATA_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "cache"
OUTPUT_DIR = Path(__file__).parent / "data"
DEFAULT_SYMBOL = "crypto:eth_usd:15min.csv"

# Config
WINDOW_SIZE = 17
TRAIN_EXAMPLES = 2000  # More examples for richer task
VAL_EXAMPLES = 400
RANDOM_SEED = 42


def compute_features(prices: np.ndarray) -> dict:
    """Compute interpretable features from price series."""
    returns = np.diff(prices) / prices[:-1] * 100

    # Trend (linear regression slope)
    x = np.arange(len(prices))
    slope = np.polyfit(x, prices, 1)[0]
    trend_pct = (slope / prices[0]) * 100 * len(prices)

    if trend_pct > 0.3:
        trend = "UPTREND"
    elif trend_pct < -0.3:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"

    # Volatility (std of returns)
    volatility = np.std(returns)
    if volatility > 0.25:
        vol_label = "HIGH"
    elif volatility > 0.12:
        vol_label = "MEDIUM"
    else:
        vol_label = "LOW"

    # Momentum (recent vs older returns)
    if len(returns) >= 8:
        recent = np.mean(returns[-4:])
        older = np.mean(returns[:4])
        momentum = recent - older
    else:
        momentum = 0

    if momentum > 0.1:
        mom_label = "ACCELERATING"
    elif momentum < -0.1:
        mom_label = "DECELERATING"
    else:
        mom_label = "STEADY"

    # Price position relative to range
    price_range = prices.max() - prices.min()
    if price_range > 0:
        position = (prices[-1] - prices.min()) / price_range
    else:
        position = 0.5

    if position > 0.8:
        pos_label = "NEAR_HIGH"
    elif position < 0.2:
        pos_label = "NEAR_LOW"
    else:
        pos_label = "MID_RANGE"

    # Simple patterns
    higher_highs = 0
    lower_lows = 0
    for i in range(2, len(prices) - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1] if i+1 < len(prices) else True:
            # Local high
            pass

    return {
        "trend": trend,
        "volatility": vol_label,
        "momentum": mom_label,
        "position": pos_label,
        "trend_pct": trend_pct,
        "vol_value": volatility,
    }


def encode_prices(prices: np.ndarray) -> str:
    """Encode prices with normalization."""
    p90 = np.percentile(np.abs(prices), 90)
    scale = p90 if p90 > 0 else 1.0
    scaled = prices / scale

    pairs = []
    for idx, val in enumerate(scaled):
        val_int = int(val * 10)
        pairs.append(f"({idx}:{abs(val_int)})")

    return "".join(pairs), scale


def format_input(prices: np.ndarray, scale: float) -> str:
    """Format input with prices AND features."""
    encoded, _ = encode_prices(prices)
    features = compute_features(prices)

    # Rich input format
    input_str = (
        f"PRICES: {encoded}\n"
        f"SCALE: {scale:.0f}\n"
        f"CURRENT: {prices[-1]:.2f}\n"
        f"TREND: {features['trend']}\n"
        f"VOLATILITY: {features['volatility']}\n"
        f"MOMENTUM: {features['momentum']}\n"
        f"POSITION: {features['position']}"
    )
    return input_str


def format_output(prices: np.ndarray, future_price: float) -> str:
    """Format rich output with multiple targets."""
    last_price = prices[-1]
    change_pct = ((future_price - last_price) / last_price) * 100

    # Direction
    if change_pct > 0.3:
        direction = "STRONG_UP"
    elif change_pct > 0.1:
        direction = "UP"
    elif change_pct < -0.3:
        direction = "STRONG_DOWN"
    elif change_pct < -0.1:
        direction = "DOWN"
    else:
        direction = "FLAT"

    # Confidence based on recent volatility
    recent_returns = np.diff(prices[-5:]) / prices[-5:-1] * 100
    vol = np.std(recent_returns) if len(recent_returns) > 1 else 0.1

    # Higher volatility = lower confidence
    if vol < 0.1:
        confidence = "HIGH"
    elif vol < 0.2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Rich output format
    output_str = (
        f"DIRECTION: {direction}\n"
        f"CHANGE: {change_pct:+.2f}%\n"
        f"NEXT_PRICE: {future_price:.2f}\n"
        f"CONFIDENCE: {confidence}"
    )
    return output_str


def get_simple_direction(change_pct: float) -> str:
    """Simple UP/DOWN for evaluation."""
    return "UP" if change_pct >= 0 else "DOWN"


def load_data(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Load OHLCV data."""
    path = DATA_PATH / symbol
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_example(df: pd.DataFrame, idx: int) -> dict | None:
    """Generate single training example with rich I/O."""
    if idx + WINDOW_SIZE + 1 >= len(df):
        return None

    hist = df.iloc[idx : idx + WINDOW_SIZE]
    future = df.iloc[idx + WINDOW_SIZE]

    hist_prices = hist["close"].values
    future_price = future["close"]

    _, scale = encode_prices(hist_prices)

    user_input = format_input(hist_prices, scale)
    model_output = format_output(hist_prices, future_price)

    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{
                "text": (
                    "You are a crypto price prediction model. "
                    "Given price data and market features, predict the next 15-minute movement. "
                    "Output DIRECTION, CHANGE%, NEXT_PRICE, and CONFIDENCE."
                )
            }]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_input}]},
            {"role": "model", "parts": [{"text": model_output}]}
        ],
        # Metadata for evaluation
        "_change_pct": ((future_price - hist_prices[-1]) / hist_prices[-1]) * 100,
        "_direction": get_simple_direction(((future_price - hist_prices[-1]) / hist_prices[-1]) * 100),
    }


def generate_dataset(df: pd.DataFrame, num_examples: int, exclude_indices: set = None) -> tuple:
    """Generate dataset with random sampling."""
    exclude_indices = exclude_indices or set()
    valid_indices = [
        i for i in range(len(df) - WINDOW_SIZE - 1)
        if i not in exclude_indices
    ]

    random.shuffle(valid_indices)
    examples = []

    for idx in valid_indices:
        if len(examples) >= num_examples:
            break
        ex = generate_example(df, idx)
        if ex:
            examples.append(ex)
            exclude_indices.add(idx)

    return examples, exclude_indices


def save_jsonl(examples: list[dict], path: Path, include_meta: bool = False) -> None:
    """Save examples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            # Remove metadata for training file
            ex_copy = {k: v for k, v in ex.items() if not k.startswith("_")}
            f.write(json.dumps(ex_copy) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def main():
    print("=" * 60)
    print("GENERATE FINE-TUNING DATA V2 (Rich I/O)")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    # Load data
    print(f"\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} bars")

    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Train pool: {len(train_df)} bars")
    print(f"Val pool: {len(val_df)} bars")

    # Generate training data
    print(f"\nGenerating {TRAIN_EXAMPLES} training examples...")
    train_examples, _ = generate_dataset(train_df, TRAIN_EXAMPLES)

    # Generate validation data
    print(f"Generating {VAL_EXAMPLES} validation examples...")
    val_examples, _ = generate_dataset(val_df, VAL_EXAMPLES)

    # Check label distribution
    train_up = sum(1 for ex in train_examples if ex["_direction"] == "UP")
    val_up = sum(1 for ex in val_examples if ex["_direction"] == "UP")

    print(f"\nLabel distribution:")
    print(f"  Train: {train_up} UP, {len(train_examples) - train_up} DOWN ({train_up/len(train_examples)*100:.1f}% UP)")
    print(f"  Val:   {val_up} UP, {len(val_examples) - val_up} DOWN ({val_up/len(val_examples)*100:.1f}% UP)")

    # Save
    save_jsonl(train_examples, OUTPUT_DIR / "train_v2.jsonl")
    save_jsonl(val_examples, OUTPUT_DIR / "val_v2.jsonl")

    # Preview
    print("\n" + "=" * 60)
    print("EXAMPLE INPUT:")
    print("=" * 60)
    print(train_examples[0]["contents"][0]["parts"][0]["text"])
    print("\n" + "=" * 60)
    print("EXAMPLE OUTPUT:")
    print("=" * 60)
    print(train_examples[0]["contents"][1]["parts"][0]["text"])

    print("\n" + "=" * 60)
    print("DONE. Next: python -m trdr.strategy.llm_predict.finetune.train_v2")
    print("=" * 60)


if __name__ == "__main__":
    main()

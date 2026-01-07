"""Generate training data for Gemini fine-tuning.

Creates JSONL files in the format required by Vertex AI supervised tuning.

Usage:
    python -m trdr.strategy.llm_predict.finetune.generate_data_v1
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
TRAIN_EXAMPLES = 1000
VAL_EXAMPLES = 200
RANDOM_SEED = 42


def encode_coordinate(series: np.ndarray) -> str:
    """Encode price series into coordinate format."""
    p90 = np.percentile(np.abs(series), 90)
    scale = p90 if p90 > 0 else 1.0
    scaled = series / scale

    pairs = []
    for idx, val in enumerate(scaled):
        val_int = int(val * 10)
        pairs.append(f"({idx}:{abs(val_int)})")

    return "".join(pairs), scale


def get_label(change_pct: float) -> str:
    """Binary label - UP or DOWN."""
    return "UP" if change_pct >= 0 else "DOWN"


def load_data(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Load OHLCV data."""
    path = DATA_PATH / symbol
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_example(df: pd.DataFrame, idx: int) -> dict | None:
    """Generate single training example."""
    if idx + WINDOW_SIZE + 1 >= len(df):
        return None

    hist = df.iloc[idx : idx + WINDOW_SIZE]
    future = df.iloc[idx + WINDOW_SIZE]

    hist_prices = hist["close"].values
    future_price = future["close"]
    last_price = hist_prices[-1]
    change_pct = ((future_price - last_price) / last_price) * 100

    encoded, scale = encode_coordinate(hist_prices)
    label = get_label(change_pct)

    # Format for Vertex AI
    user_input = f"{encoded}|{scale:.0f}|{last_price:.0f}"

    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": "Predict crypto price direction. Input: encoded prices|scale|current_price. Output: UP or DOWN."}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_input}]},
            {"role": "model", "parts": [{"text": label}]}
        ]
    }


def generate_dataset(df: pd.DataFrame, num_examples: int, exclude_indices: set = None) -> list[dict]:
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


def save_jsonl(examples: list[dict], path: Path) -> None:
    """Save examples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def main():
    print("=" * 60)
    print("GENERATE FINE-TUNING DATA")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = load_data()
    print(f"Loaded {len(df)} bars")

    # Split: use first 80% for training pool, last 20% for validation pool
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
    train_up = sum(1 for ex in train_examples if ex["contents"][1]["parts"][0]["text"] == "UP")
    val_up = sum(1 for ex in val_examples if ex["contents"][1]["parts"][0]["text"] == "UP")

    print(f"\nLabel distribution:")
    print(f"  Train: {train_up} UP, {len(train_examples) - train_up} DOWN ({train_up/len(train_examples)*100:.1f}% UP)")
    print(f"  Val:   {val_up} UP, {len(val_examples) - val_up} DOWN ({val_up/len(val_examples)*100:.1f}% UP)")

    # Save
    save_jsonl(train_examples, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_examples, OUTPUT_DIR / "val.jsonl")

    # Preview
    print("\n[PREVIEW - First training example]")
    print(json.dumps(train_examples[0], indent=2))

    print("\n" + "=" * 60)
    print("DONE. Next steps:")
    print("1. Upload to GCS: gsutil cp data/*.jsonl gs://YOUR_BUCKET/finetune/")
    print("2. Run: python -m trdr.strategy.llm_predict.finetune.train_v1")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Generate training data for Gemini fine-tuning.

Creates JSONL files in the format required by Vertex AI supervised tuning.

Usage:
    python -m trdr.strategy.llm_predict.finetune.generate_data_v3
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from trdr.strategy.llm_predict.experiment import (
    encode_sax_enhanced,
    get_triple_barrier_label,
)

# Paths
DATA_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "cache"
OUTPUT_DIR = Path(__file__).parent / "data"
DEFAULT_SYMBOL = "crypto:eth_usd:15min.csv"

# Config
WINDOW_SIZE = 17
HORIZON = 6
BARRIER_UP = 0.15
BARRIER_DOWN = 0.15
SAX_SEGMENTS = 5
SAX_ALPHABET = 5
SAX_INCLUDE_MOMENTUM = True
SAX_INCLUDE_RANGE = True
SAX_INCLUDE_EXTREMES = True
SAX_INCLUDE_RSI = True
SAX_RSI_PERIOD = 14
SAX_USE_TRIANGLES = True
SAX_USE_RANGE_SYMBOLS = True
SAX_USE_RSI_SYMBOLS = True
TRAIN_EXAMPLES = 1000
VAL_EXAMPLES = 200
RANDOM_SEED = 42


def load_data(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Load OHLCV data."""
    path = DATA_PATH / symbol
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_example(df: pd.DataFrame, idx: int) -> dict | None:
    """Generate single training example."""
    if idx + WINDOW_SIZE + HORIZON >= len(df):
        return None

    hist = df.iloc[idx : idx + WINDOW_SIZE]
    future_window = df.iloc[idx + WINDOW_SIZE : idx + WINDOW_SIZE + HORIZON]

    hist_prices = hist["close"].values
    last_price = hist_prices[-1]

    encoded, scale = encode_sax_enhanced(
        hist_prices,
        SAX_SEGMENTS,
        SAX_ALPHABET,
        SAX_INCLUDE_MOMENTUM,
        SAX_INCLUDE_RANGE,
        SAX_INCLUDE_EXTREMES,
        SAX_INCLUDE_RSI,
        SAX_RSI_PERIOD,
        use_triangles=SAX_USE_TRIANGLES,
        use_range_symbols=SAX_USE_RANGE_SYMBOLS,
        use_rsi_symbols=SAX_USE_RSI_SYMBOLS,
    )
    label, _ = get_triple_barrier_label(
        hist_prices,
        future_window["close"].values,
        BARRIER_UP,
        BARRIER_DOWN,
    )

    # Format for Vertex AI
    user_input = f"{encoded}|{scale:.0f}|{last_price:.0f}"

    return {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": (
                        "Predict crypto price direction. "
                        "Input: SAX-encoded prices|scale|current_price. "
                        "Output: UP, DOWN, or same."
                    )
                }
            ],
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
    def label_counts(examples: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {"UP": 0, "DOWN": 0, "same": 0}
        for ex in examples:
            label = ex["contents"][1]["parts"][0]["text"]
            counts[label] = counts.get(label, 0) + 1
        return counts

    train_counts = label_counts(train_examples)
    val_counts = label_counts(val_examples)

    print(f"\nLabel distribution:")
    print(
        "  Train: "
        f"UP={train_counts['UP']}, DOWN={train_counts['DOWN']}, same={train_counts['same']}"
    )
    print(
        "  Val:   "
        f"UP={val_counts['UP']}, DOWN={val_counts['DOWN']}, same={val_counts['same']}"
    )

    # Save
    save_jsonl(train_examples, OUTPUT_DIR / "train_v3.jsonl")
    save_jsonl(val_examples, OUTPUT_DIR / "val_v3.jsonl")

    # Preview
    print("\n[PREVIEW - First training example]")
    print(json.dumps(train_examples[0], indent=2))

    print("\n" + "=" * 60)
    print("DONE. Next steps:")
    print("1. Upload to GCS: gsutil cp data/*_v3.jsonl gs://YOUR_BUCKET/finetune/")
    print("2. Run: python -m trdr.strategy.llm_predict.finetune.train_v3")
    print("=" * 60)


if __name__ == "__main__":
    main()

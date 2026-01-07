"""Test the fine-tuned model.

Usage:
    python -m trdr.strategy.llm_predict.finetune.test_v3
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trdr.strategy.llm_predict.experiment import (
    encode_sax_enhanced,
    get_triple_barrier_label,
)
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent.parent.parent / ".env")

# Paths
DATA_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "cache"
DEFAULT_SYMBOL = "crypto:eth_usd:15min.csv"
CONFIG_PATH = Path(__file__).parent / "tuned_model_v3.txt"

# Test config
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
NUM_TESTS = 40


def load_tuned_model_config() -> tuple[str, str]:
    """Load tuned model config from file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Tuned model config not found at {CONFIG_PATH}. "
            "Run train_v3.py first to fine-tune the model."
        )

    config = {}
    with open(CONFIG_PATH) as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            config[key] = val

    return config.get("MODEL_NAME"), config.get("ENDPOINT_NAME")


def load_data(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Load OHLCV data."""
    path = DATA_PATH / symbol
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


class TunedModelPredictor:
    """Predictor using fine-tuned Vertex AI model."""

    def __init__(self, model_name: str, endpoint_name: str):
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        vertexai.init(project=project_id, location="us-central1")

        # Use endpoint name for tuned models
        self.model = GenerativeModel(endpoint_name)
        print(f"[TunedModel] Loaded endpoint: {endpoint_name}")
        self.debug_raw = True
        self.debug_limit = 5
        self.debug_count = 0

    def predict(self, encoded: str, scale: float, price: float) -> str:
        """Make prediction."""
        user_input = f"{encoded}|{scale:.0f}|{price:.0f}"

        for attempt in range(3):
            try:
                response = self.model.generate_content(
                    user_input,
                    generation_config={"max_output_tokens": 10}
                )
                text = response.text.strip().upper()
                if self.debug_raw and self.debug_count < self.debug_limit:
                    self.debug_count += 1
                    print(f"RAW[{self.debug_count}]: {response.text.strip()}")

                # Parse response
                if "SAME" in text:
                    return "same"
                if "UP" in text:
                    return "UP"
                elif "DOWN" in text:
                    return "DOWN"
                else:
                    return "same"  # Default fallback

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"[Rate limit] Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        return "same"  # Fallback


def run_test(predictor: TunedModelPredictor, test_df: pd.DataFrame):
    """Run test predictions."""
    print("\n" + "=" * 60)
    print(f"TESTING FINE-TUNED MODEL ({NUM_TESTS} predictions)")
    print("=" * 60)

    correct = 0
    total = 0
    labels = ["UP", "DOWN", "same"]
    confusion = {a: {p: 0 for p in labels} for a in labels}
    returns = []

    step = max(1, len(test_df) // NUM_TESTS)

    for i in range(WINDOW_SIZE, len(test_df) - 1, step):
        if total >= NUM_TESTS:
            break

        hist = test_df.iloc[i - WINDOW_SIZE : i]
        future_window = test_df.iloc[i : i + HORIZON]

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
        if total == 0:
            print(f"TEST INPUT: {encoded}|{scale:.0f}|{last_price:.0f}")
            if "|M:" not in encoded or "|R:" not in encoded or "|RSI:" not in encoded:
                print("WARN: Missing expected SAX symbols in test encoding.")
        actual, change_pct = get_triple_barrier_label(
            hist_prices,
            future_window["close"].values,
            BARRIER_UP,
            BARRIER_DOWN,
        )

        pred = predictor.predict(encoded, scale, last_price)

        is_correct = pred == actual
        if is_correct:
            correct += 1
        total += 1
        confusion[actual][pred] += 1
        if pred == "UP":
            returns.append(change_pct)
        elif pred == "DOWN":
            returns.append(-change_pct)
        else:
            returns.append(0.0)

        status = "+" if is_correct else "x"
        print(f"[{total:2}] {status} Pred: {pred:4} | Actual: {actual:4} | Δ: {change_pct:+.2f}%")

        time.sleep(0.3)  # Rate limit

    # Summary
    accuracy = correct / total if total > 0 else 0
    print("\n" + "-" * 60)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1%})")
    for label in labels:
        pred_total = sum(confusion[a][label] for a in labels)
        actual_total = sum(confusion[label].values())
        tp = confusion[label][label]
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / actual_total if actual_total else 0.0
        print(f"{label} precision: {precision:.1%}, recall: {recall:.1%}")
    avg_return = sum(returns) / len(returns) if returns else 0.0
    print(f"Avg return per prediction: {avg_return:+.3f}%")
    print("-" * 60)

    # Compare to baseline (50% random)
    if accuracy > 0.55:
        print("BETTER than random baseline (50%)")
    elif accuracy < 0.45:
        print("WORSE than random baseline (50%)")
    else:
        print("~Same as random baseline (50%)")

    return accuracy


def main():
    print("=" * 60)
    print("TEST FINE-TUNED MODEL")
    print("=" * 60)

    # Load tuned model config
    try:
        model_name, endpoint_name = load_tuned_model_config()
        print(f"\nModel:    {model_name}")
        print(f"Endpoint: {endpoint_name}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # Load test data (use last 10% of data - unseen during training)
    print("\nLoading test data...")
    df = load_data()
    test_start = int(len(df) * 0.9)  # Last 10%
    test_df = df.iloc[test_start:].reset_index(drop=True)
    print(f"Test data: {len(test_df)} bars (last 10% of dataset)")

    # Initialize predictor
    try:
        predictor = TunedModelPredictor(model_name, endpoint_name)
    except Exception as e:
        print(f"\nERROR initializing model: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you're authenticated: gcloud auth application-default login")
        print("  2. Install: pip install google-cloud-aiplatform")
        return

    # Run test
    accuracy = run_test(predictor, test_df)

    # Save results
    results_path = Path(__file__).parent / "test_results.txt"
    with open(results_path, "a") as f:
        f.write(f"version=v3,accuracy={accuracy:.4f},n={NUM_TESTS},model={model_name}\n")
    print(f"\nResults appended to {results_path}")


if __name__ == "__main__":
    main()

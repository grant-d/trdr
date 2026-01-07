"""Test the V2 fine-tuned model with rich I/O.

Usage:
    python -m trdr.strategy.llm_predict.finetune.test_v2
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent.parent.parent / ".env")

# Paths
DATA_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "cache"
DEFAULT_SYMBOL = "crypto:eth_usd:15min.csv"
CONFIG_PATH = Path(__file__).parent / "tuned_model_v2.txt"

# Test config
WINDOW_SIZE = 17
NUM_TESTS = 50


def load_tuned_model_config() -> tuple[str, str]:
    """Load tuned model config from file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Tuned model config not found at {CONFIG_PATH}. "
            "Run train.py first to fine-tune the model."
        )

    config = {}
    with open(CONFIG_PATH) as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            config[key] = val

    return config.get("MODEL_NAME"), config.get("ENDPOINT_NAME")


def encode_prices(prices: np.ndarray) -> tuple[str, float]:
    """Encode prices with normalization."""
    p90 = np.percentile(np.abs(prices), 90)
    scale = p90 if p90 > 0 else 1.0
    scaled = prices / scale

    pairs = []
    for idx, val in enumerate(scaled):
        val_int = int(val * 10)
        pairs.append(f"({idx}:{abs(val_int)})")

    return "".join(pairs), scale


def compute_features(prices: np.ndarray) -> dict:
    """Compute interpretable features from price series."""
    returns = np.diff(prices) / prices[:-1] * 100

    # Trend
    x = np.arange(len(prices))
    slope = np.polyfit(x, prices, 1)[0]
    trend_pct = (slope / prices[0]) * 100 * len(prices)

    if trend_pct > 0.3:
        trend = "UPTREND"
    elif trend_pct < -0.3:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"

    # Volatility
    volatility = np.std(returns)
    if volatility > 0.25:
        vol_label = "HIGH"
    elif volatility > 0.12:
        vol_label = "MEDIUM"
    else:
        vol_label = "LOW"

    # Momentum
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

    # Position
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

    return {
        "trend": trend,
        "volatility": vol_label,
        "momentum": mom_label,
        "position": pos_label,
    }


def format_input(prices: np.ndarray, scale: float) -> str:
    """Format input with prices AND features (V2 format)."""
    encoded, _ = encode_prices(prices)
    features = compute_features(prices)

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


def parse_v2_output(text: str) -> dict:
    """Parse V2 rich output format."""
    result = {
        "direction": None,
        "change_pct": None,
        "next_price": None,
        "confidence": None,
    }

    # Try to parse each field
    dir_match = re.search(r"DIRECTION:\s*(STRONG_UP|UP|FLAT|DOWN|STRONG_DOWN)", text, re.I)
    if dir_match:
        result["direction"] = dir_match.group(1).upper()

    change_match = re.search(r"CHANGE:\s*([+-]?\d+\.?\d*)%", text)
    if change_match:
        result["change_pct"] = float(change_match.group(1))

    price_match = re.search(r"NEXT_PRICE:\s*(\d+\.?\d*)", text)
    if price_match:
        result["next_price"] = float(price_match.group(1))

    conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", text, re.I)
    if conf_match:
        result["confidence"] = conf_match.group(1).upper()

    return result


def get_simple_direction(parsed: dict, change_pct: float) -> str:
    """Extract simple UP/DOWN from parsed output."""
    if parsed["direction"]:
        d = parsed["direction"]
        if d in ("STRONG_UP", "UP"):
            return "UP"
        elif d in ("STRONG_DOWN", "DOWN"):
            return "DOWN"
        else:  # FLAT
            # Use change_pct from output if available
            if parsed["change_pct"] is not None:
                return "UP" if parsed["change_pct"] >= 0 else "DOWN"
            return "UP"  # Default
    return "UP"  # Fallback


def get_actual_label(change_pct: float) -> str:
    """Binary label from actual change."""
    return "UP" if change_pct >= 0 else "DOWN"


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

        self.model = GenerativeModel(endpoint_name)
        print(f"[TunedModel] Loaded endpoint: {endpoint_name}")

    def predict(self, user_input: str) -> dict:
        """Make prediction and return parsed result."""
        for attempt in range(3):
            try:
                response = self.model.generate_content(
                    user_input,
                    generation_config={"max_output_tokens": 100}
                )
                text = response.text.strip()
                return parse_v2_output(text), text

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"[Rate limit] Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        return {"direction": None}, ""


def run_test(predictor: TunedModelPredictor, test_df: pd.DataFrame):
    """Run test predictions."""
    print("\n" + "=" * 60)
    print(f"TESTING V2 FINE-TUNED MODEL ({NUM_TESTS} predictions)")
    print("=" * 60)

    correct = 0
    total = 0
    predictions = {"UP": 0, "DOWN": 0}
    actuals = {"UP": 0, "DOWN": 0}
    labels = ["UP", "DOWN"]
    confusion = {a: {p: 0 for p in labels} for a in labels}
    returns = []

    step = max(1, len(test_df) // NUM_TESTS)

    for i in range(WINDOW_SIZE, len(test_df) - 1, step):
        if total >= NUM_TESTS:
            break

        hist = test_df.iloc[i - WINDOW_SIZE : i]
        future = test_df.iloc[i]

        hist_prices = hist["close"].values
        future_price = future["close"]
        last_price = hist_prices[-1]
        change_pct = ((future_price - last_price) / last_price) * 100

        _, scale = encode_prices(hist_prices)
        user_input = format_input(hist_prices, scale)
        actual = get_actual_label(change_pct)
        actuals[actual] += 1

        parsed, raw_text = predictor.predict(user_input)
        pred = get_simple_direction(parsed, change_pct)
        predictions[pred] += 1

        is_correct = pred == actual
        if is_correct:
            correct += 1
        total += 1
        confusion[actual][pred] += 1
        returns.append(change_pct if pred == "UP" else -change_pct)

        status = "+" if is_correct else "x"

        # Show rich output
        pred_change = f"{parsed['change_pct']:+.2f}%" if parsed['change_pct'] else "?"
        pred_conf = parsed['confidence'] or "?"
        print(f"[{total:2}] {status} Pred: {pred:4} ({pred_change}, {pred_conf}) | Actual: {actual:4} ({change_pct:+.2f}%)")

        time.sleep(0.3)  # Rate limit

    # Summary
    accuracy = correct / total if total > 0 else 0
    print("\n" + "-" * 60)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Predictions: UP={predictions['UP']}, DOWN={predictions['DOWN']}")
    print(f"Actuals:     UP={actuals['UP']}, DOWN={actuals['DOWN']}")
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

    # Compare to baseline
    if accuracy > 0.55:
        print("BETTER than random baseline (50%)")
    elif accuracy < 0.45:
        print("WORSE than random baseline (50%)")
    else:
        print("~Same as random baseline (50%)")

    return accuracy


def main():
    print("=" * 60)
    print("TEST V2 FINE-TUNED MODEL (Rich I/O)")
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
    test_start = int(len(df) * 0.9)
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
        f.write(f"version=v2,accuracy={accuracy:.4f},n={NUM_TESTS},model={model_name}\n")
    print(f"\nResults appended to {results_path}")


if __name__ == "__main__":
    main()

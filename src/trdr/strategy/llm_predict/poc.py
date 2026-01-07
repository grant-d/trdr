"""LLM-based price prediction PoC using Gemini and OpenAI.

Uses coordinate-indexed encoding to represent price history as tokens.
Predicts direction: UP|up|same|down|DOWN (5 levels based on % change).

Usage:
    export GOOGLE_API_KEY="your-api-key"
    export OPENAI_API_KEY="your-api-key"
    python -m trdr.strategy.llm_predict.poc
"""

import enum
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent.parent.parent / ".env")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "cache"
DEFAULT_SYMBOL = "crypto:eth_usd:15min.csv"

# gemini-2.5-flash-lite, gemini-3-flash-preview, gemini-3-pro-preview
GEMINI_MODEL = "gemini-3-flash-preview"
# gpt-5.1-codex-mini, gpt-5-mini, gpt-5.2, gpt-5.2-pro (legacy: gpt-4o-mini)
OPENAI_MODEL = "gpt-5-mini" 
# claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5
CLAUDE_MODEL = "claude-haiku-4-5" 

WINDOW_SIZE = 17  # Fixed window size
TRAINING_EXAMPLES = 300
PREDICTION_INTERVAL = 4


class Direction(enum.Enum):
    """Price direction prediction (5 levels)."""

    UP = "UP"  # > +0.3%
    up = "up"  # +0.15% to +0.3%
    same = "same"  # -0.15% to +0.15%
    down = "down"  # -0.3% to -0.15%
    DOWN = "DOWN"  # < -0.3%


class PredictionResult(BaseModel):
    """Structured prediction result."""

    direction: Direction
    change: float
    reason: str


# =============================================================================
# ENCODING
# =============================================================================


def encode_coordinate(series: np.ndarray, precision: int = 1) -> tuple[str, float]:
    """Encode price series into coordinate format.

    Compact format without spaces: (0:9)(1:9)(2:10)...
    """
    p90 = np.percentile(np.abs(series), 90)
    scale = p90 if p90 > 0 else 1.0
    scaled = series / scale

    pairs = []
    for idx, val in enumerate(scaled):
        val_int = int(val * (10**precision))
        pairs.append(f"({idx}:{abs(val_int)})")

    return "".join(pairs), scale


# =============================================================================
# DIRECTION LABELING
# =============================================================================


def get_direction(change_pct: float) -> str:
    """Convert % change to direction label."""
    if change_pct > 0.3:
        return "UP"
    elif change_pct > 0.15:
        return "up"
    elif change_pct < -0.3:
        return "DOWN"
    elif change_pct < -0.15:
        return "down"
    else:
        return "same"


def create_label(hist_prices: np.ndarray, future_price: float) -> tuple[str, float]:
    """Create direction label from price movement.

    Returns:
        Tuple of (direction, change_percent)
    """
    last_price = hist_prices[-1]
    change_pct = ((future_price - last_price) / last_price) * 100
    direction = get_direction(change_pct)
    return direction, change_pct


# =============================================================================
# TRAINING DATA
# =============================================================================


def generate_training_examples(df: pd.DataFrame, num_examples: int = 200) -> list[dict]:
    """Generate training examples from historical data."""
    examples = []
    step_size = max(1, len(df) // num_examples)

    for i in range(0, len(df) - WINDOW_SIZE - 1, step_size):
        if len(examples) >= num_examples:
            break

        if i + WINDOW_SIZE + 1 >= len(df):
            continue

        hist = df.iloc[i : i + WINDOW_SIZE]
        future = df.iloc[i + WINDOW_SIZE]

        hist_prices = hist["close"].values
        future_price = future["close"]

        encoded, scale = encode_coordinate(hist_prices, precision=1)
        direction, change_pct = create_label(hist_prices, future_price)

        examples.append(
            {
                "encoded": encoded,
                "scale": scale,
                "last_price": hist_prices[-1],
                "direction": direction,
                "change_pct": change_pct,
            }
        )

    return examples


SYSTEM_PROMPT = """You are a professional cryptocurrency trading analyst specializing in pattern recognition from coordinate-encoded time series data.

## YOUR TASK
Analyze 15-minute price movement predictions using coordinate-indexed candle patterns.

## DATA FORMAT
- TIME SERIES: (coordinate:scaled_value) pairs
- SCALE: 90th percentile normalization
- PRICE: Current price
- WINDOW: 17 timesteps

## PATTERN RULES FROM TRAINING EXAMPLES

### 1. Count the "10" values (spikes)
- 0-1 spikes: Usually HOLD
- 2+ consecutive spikes: Usually SELL
- Single spike: Depends on LOCATION

### 2. Location hierarchy (strongest → weakest)
- Positions 0-3 (early): Strong signal
- Positions 4-7 (mid): Medium signal
- Positions 8-11 (late): Weak signal
- Positions 12-16 (terminal): Almost no signal

### 3. Early spike (pos 0-2) patterns:
- Position 0 = 10: BUY bias
- Position 1 = 10: Mixed—check scale
- Position 2 = 10: Often SELL

### 4. Scale context (volatility):
- Scale > 4000: High volatility → stronger signals, higher confidence
- Scale 2000-4000: Moderate
- Scale < 1500: Low volatility → HOLD more likely

### 5. GOLDEN RULE
Never decide on single spike position alone. Always check:
1. Total spike count?
2. Clustered or scattered?
3. What's the scale?
4. Price context?

## Critical Signals

SELL (HIGH):
- 2+ consecutive spikes mid-sequence
- Multiple clustered 10s
- Early spike (>3500 scale) with downside

BUY (HIGH):
- Early spike (>3000 scale) followed by 9s
- Scattered 10s

HOLD (LOW):
- Mostly 9s
- Late spikes only
- Conflicting signals

## Critical DON'Ts:
- ❌ Decide on position alone
- ❌ Ignore scale
- ❌ Assume position 0 = SELL (it's actually BUY bias!)
- ❌ Over-interpret single isolated spikes
- ❌ Ignore price context

## DIRECTIONS
- UP: > +0.3%
- up: +0.15% to +0.3%
- same: -0.15% to +0.15%
- down: -0.3% to -0.15%
- DOWN: < -0.3%

## RESPONSE FORMAT
DECISION|CONFIDENCE|PREDICTION
Example: UP|HIGH|+0.55 or down|LOW|-0.20 or same|MEDIUM|+0.05
"""


def format_training_context(examples: list[dict]) -> str:
    """Format training examples for LLM context."""
    lines = [SYSTEM_PROMPT, "", "## TRAINING EXAMPLES", ""]

    for i, ex in enumerate(examples):
        line = f"#{i+1} {ex['encoded']}|{ex['scale']:.0f}|{ex['last_price']:.0f}→{ex['direction']}|{ex['change_pct']:+.2f}"
        lines.append(line)

    return "\n".join(lines)


# =============================================================================
# GEMINI PREDICTOR
# =============================================================================


class GeminiPredictor:
    """Gemini-based price predictor using implicit caching."""

    def __init__(self, training_context: str):
        from google import genai

        self.client = genai.Client()
        self.model = GEMINI_MODEL
        self._training_context = training_context

    def gemini_start_session(self) -> None:
        """Initialize Gemini predictor."""
        print(f"[Gemini] Ready with {len(self._training_context)} chars context (implicit caching)")

    def gemini_predict(self, encoded: str, scale: float, price: float) -> PredictionResult:
        """Make prediction using Gemini with full context each time."""
        from google.genai import types

        query = f"{encoded}|{scale:.0f}|{price:.0f}"

        prompt = f"""{self._training_context}

## TEST
{query}"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=200,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = response.text.strip() if response.text else ""

        return self._parse_response(text)

    def _parse_response(self, text: str) -> PredictionResult:
        """Parse DECISION|CONFIDENCE|PREDICTION response."""
        lines = text.strip().split("\n")
        pred_line = ""

        # Find line with pipe containing direction
        for line in lines:
            if "|" in line:
                for d in ["UP", "up", "same", "down", "DOWN"]:
                    if d in line:
                        pred_line = line.strip()
                        break
                if pred_line:
                    break

        parts = pred_line.split("|")
        try:
            direction = Direction(parts[0].strip())
            # parts[1] is confidence, parts[2] is change
            change = float(parts[2].strip().replace("%", "").replace("+", "")) if len(parts) > 2 else 0.0
        except (IndexError, ValueError, KeyError):
            direction = Direction.same
            change = 0.0

        return PredictionResult(direction=direction, change=change, reason=pred_line)

    def gemini_cleanup(self) -> None:
        """Cleanup Gemini predictor."""
        print("[Gemini] Done")


# =============================================================================
# OPENAI PREDICTOR
# =============================================================================


class OpenAIPredictor:
    """OpenAI-based price predictor supporting chat, responses, and completions APIs."""

    def __init__(self, training_context: str):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = OPENAI_MODEL
        self.messages = []
        self._training_context = training_context
        # Detect API type based on model
        # codex and pro models use responses API, gpt-5.2 uses chat
        self._use_responses = "codex" in self.model or "pro" in self.model

    def openai_start_session(self) -> None:
        """Start OpenAI session by loading training context."""
        api_type = "responses" if self._use_responses else "chat"
        print(f"[OpenAI] Starting session with {len(self._training_context)} chars ({api_type} API)...")

        self.messages = [{"role": "system", "content": self._training_context}]
        print("[OpenAI] Session ready")

    def openai_predict(self, encoded: str, scale: float, price: float) -> PredictionResult:
        """Make prediction using appropriate OpenAI API."""
        query = f"## TEST\n{encoded}|{scale:.0f}|{price:.0f}"

        if self._use_responses:
            return self._predict_responses(query)
        else:
            return self._predict_chat(query)

    def _predict_chat(self, query: str) -> PredictionResult:
        """Use chat completions API."""
        self.messages.append({"role": "user", "content": query})
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "max_completion_tokens": 100,
        }
        # gpt-5-mini doesn't support custom temperature
        if "5-mini" not in self.model:
            kwargs["temperature"] = 0.1
        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": text})
        return self._parse_response(text)

    def _predict_responses(self, query: str) -> PredictionResult:
        """Use responses API for codex models."""
        self.messages.append({"role": "user", "content": query})
        response = self.client.responses.create(
            model=self.model,
            input=self.messages,
            max_output_tokens=100,
        )
        text = response.output_text.strip() if response.output_text else ""
        self.messages.append({"role": "assistant", "content": text})
        return self._parse_response(text)

    def _parse_response(self, text: str) -> PredictionResult:
        """Parse DECISION|CONFIDENCE|PREDICTION response."""
        lines = text.strip().split("\n")
        pred_line = ""

        # Find line with pipe containing direction
        for line in lines:
            if "|" in line:
                for d in ["UP", "up", "same", "down", "DOWN"]:
                    if d in line:
                        pred_line = line.strip()
                        break
                if pred_line:
                    break

        parts = pred_line.split("|")
        try:
            direction = Direction(parts[0].strip())
            change = float(parts[2].strip().replace("%", "").replace("+", "")) if len(parts) > 2 else 0.0
        except (IndexError, ValueError, KeyError):
            direction = Direction.same
            change = 0.0

        return PredictionResult(direction=direction, change=change, reason=pred_line)

    def openai_cleanup(self) -> None:
        """Cleanup OpenAI session."""
        self.messages = []
        print("[OpenAI] Session closed")


# =============================================================================
# CLAUDE PREDICTOR
# =============================================================================


class ClaudePredictor:
    """Claude-based price predictor using Anthropic API with prompt caching."""

    def __init__(self, training_context: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = CLAUDE_MODEL
        self._training_context = training_context

    def claude_start_session(self) -> None:
        """Initialize Claude predictor."""
        print(f"[Claude] Ready with {len(self._training_context)} chars context (prompt caching)")

    def claude_predict(self, encoded: str, scale: float, price: float) -> PredictionResult:
        """Make prediction using Claude with prompt caching."""
        query = f"{encoded}|{scale:.0f}|{price:.0f}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            system=[
                {
                    "type": "text",
                    "text": self._training_context,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": f"## TEST\n{query}"}],
        )
        text = response.content[0].text.strip() if response.content else ""

        return self._parse_response(text)

    def _parse_response(self, text: str) -> PredictionResult:
        """Parse DECISION|CONFIDENCE|PREDICTION response."""
        lines = text.strip().split("\n")
        pred_line = ""

        for line in lines:
            if "|" in line:
                for d in ["UP", "up", "same", "down", "DOWN"]:
                    if d in line:
                        pred_line = line.strip()
                        break
                if pred_line:
                    break

        parts = pred_line.split("|")
        try:
            direction = Direction(parts[0].strip())
            change = float(parts[2].strip().replace("%", "").replace("+", "")) if len(parts) > 2 else 0.0
        except (IndexError, ValueError, KeyError):
            direction = Direction.same
            change = 0.0

        return PredictionResult(direction=direction, change=change, reason=pred_line)

    def claude_cleanup(self) -> None:
        """Cleanup Claude predictor."""
        print("[Claude] Done")


# =============================================================================
# MAIN RUNNER
# =============================================================================


def load_data(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Load OHLCV data from cache."""
    path = DATA_PATH / symbol
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def run_test(predictor, predict_fn, name: str, test_df: pd.DataFrame, num_tests: int = 20):
    """Run predictions and collect results."""
    print(f"\n{'='*60}")
    print(f"{name} - Running {num_tests} predictions")
    print("=" * 60)

    correct = 0
    total = 0
    results = []

    for i in range(WINDOW_SIZE, len(test_df) - 1, PREDICTION_INTERVAL):
        if i + 1 >= len(test_df):
            break

        hist = test_df.iloc[i - WINDOW_SIZE : i]
        future = test_df.iloc[i]

        hist_prices = hist["close"].values
        future_price = future["close"]

        encoded, scale = encode_coordinate(hist_prices, precision=1)
        last_price = hist_prices[-1]

        pred = predict_fn(encoded, scale, last_price)
        actual_dir, actual_change = create_label(hist_prices, future_price)

        is_correct = pred.direction.value == actual_dir
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "predicted": pred.direction.value,
            "actual": actual_dir,
            "correct": is_correct,
            "pred_change": pred.change,
            "actual_change": actual_change,
        })

        status = "+" if is_correct else "x"
        print(f"[{total}] {status} Pred: {pred.direction.value:4} | Actual: {actual_dir:4} | Change: {actual_change:+.2f}%")

        if total >= num_tests:
            break

        time.sleep(0.3)

    # Summary
    print("-" * 60)
    accuracy = correct / total if total > 0 else 0
    print(f"Results: {correct}/{total} correct ({accuracy:.1%})")

    # Direction accuracy (non-same)
    directional = [r for r in results if r["actual"] != "same"]
    if directional:
        dir_correct = sum(1 for r in directional if r["correct"])
        print(f"Directional: {dir_correct}/{len(directional)} ({dir_correct/len(directional):.1%})")

    return results


def run_poc():
    """Run the LLM prediction PoC."""
    print("=" * 60)
    print("LLM DIRECTION PREDICTION POC")
    print("=" * 60)

    # Load data
    print(f"\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} bars")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Generate training
    print(f"\nGenerating {TRAINING_EXAMPLES} training examples...")
    examples = generate_training_examples(train_df, TRAINING_EXAMPLES)
    context = format_training_context(examples)
    print(f"Context: {len(context)} chars")

    # Preview
    print("\n[CONTEXT PREVIEW]")
    for line in context.split("\n")[:5]:
        print(f"  {line}")

    # Test Gemini
    if os.environ.get("GOOGLE_API_KEY"):
        gemini = GeminiPredictor(context)
        try:
            gemini.gemini_start_session()
            run_test(gemini, gemini.gemini_predict, "GEMINI", test_df)
        finally:
            gemini.gemini_cleanup()
    else:
        print("\n[Gemini] GOOGLE_API_KEY not set, skipping")

    # # Test OpenAI
    # if os.environ.get("OPENAI_API_KEY"):
    #     openai_pred = OpenAIPredictor(context)
    #     try:
    #         openai_pred.openai_start_session()
    #         run_test(openai_pred, openai_pred.openai_predict, "OPENAI", test_df)
    #     finally:
    #         openai_pred.openai_cleanup()
    # else:
    #     print("\n[OpenAI] OPENAI_API_KEY not set, skipping")

    # # Test Claude
    # if os.environ.get("ANTHROPIC_API_KEY"):
    #     claude = ClaudePredictor(context)
    #     try:
    #         claude.claude_start_session()
    #         run_test(claude, claude.claude_predict, "CLAUDE", test_df)
    #     finally:
    #         claude.claude_cleanup()
    # else:
    #     print("\n[Claude] ANTHROPIC_API_KEY not set, skipping")


if __name__ == "__main__":
    run_poc()

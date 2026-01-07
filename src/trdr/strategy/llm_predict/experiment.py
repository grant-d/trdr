"""LLM prediction experiments - systematic parameter testing."""

import enum
import time
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent.parent / ".env")

DATA_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "cache"
DEFAULT_SYMBOL = "crypto:eth_usd:15min.csv"


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    name: str
    model: str = "gemini-3-flash-preview"
    symbol: str = DEFAULT_SYMBOL
    window_size: int = 17
    training_examples: int = 300
    training_stride: int | None = None
    retrieval_k: int = 16
    use_retrieval_examples: bool = True
    num_tests: int = 40
    num_folds: int = 1
    label_type: str = "triple_barrier"  # direction5, direction3, binary, price_pct, triple_barrier, hold_bucket, fixed_hold, trailing_barrier
    encoding_type: str = "sax"  # coordinate, signed_coordinate, rank, delta_bins, multichannel, sax, sax_rsi,
    # returns, log_returns, sax_log_returns, gauss_returns, gauss_log_returns
    horizon: int = 6
    barrier_up: float = 0.15
    barrier_down: float = 0.15
    hold_buckets: tuple[int, int, int] = (4, 6, 8)
    sax_paa_segments: int = 5
    sax_alphabet_size: int = 5
    sax_include_momentum: bool = True
    sax_include_range: bool = True
    sax_include_extremes: bool = True
    sax_include_rsi: bool = True
    sax_rsi_period: int = 14
    sax_use_arrows: bool = False
    sax_use_triangles: bool = True
    sax_use_range_symbols: bool = True
    sax_use_rsi_symbols: bool = True
    sax_potential_top_n: int | None = None
    sax_potential_horizon: int = 24
    sax_potential_bucket_count: int | None = None
    sax_potential_up_only: bool = False
    sax_future_shape: bool = False
    sax_future_shape_horizon: int = 12
    sax_future_shape_segments: int = 5
    sax_prospect_buckets: bool = False
    sax_prospect_horizon: int = 12
    sax_prospect_bucket_count: int = 3
    sax_prospect_time: bool = False
    sax_gate_top_n: int | None = None
    sax_include_occurrence: bool = False
    htf_symbol: str | None = None
    htf_rsi_period: int = 14
    use_cached_content: bool = False
    cache_ttl: str = "300s"
    use_retrieval: bool = True
    use_sax_retrieval: bool = False
    add_patch_summary: bool = False
    patch_size: int = 4
    require_summary: bool = False
    regime_min_atr_pct: float | None = None
    atr_lookback: int = 14
    add_index_channel: bool = False
    use_chat: bool = False  # chat session vs single requests
    temperature: float | None = 0.0  # None = model default
    top_p: float | None = None  # nucleus sampling
    top_k: int | None = None  # top-k sampling
    prompt_style: str = "multi_vote"  # minimal, detailed, cot, multi_answer

    def __str__(self) -> str:
        extras = []
        if self.encoding_type != "coordinate":
            extras.append(f"enc={self.encoding_type}")
        if self.encoding_type in {"coordinate", "signed_coordinate"} and self.add_index_channel:
            extras.append("idx=1")
        if self.symbol != DEFAULT_SYMBOL:
            extras.append(f"sym={self.symbol}")
        if self.encoding_type == "sax":
            extras.append(f"sax_p={self.sax_paa_segments}")
            extras.append(f"sax_a={self.sax_alphabet_size}")
            if not self.sax_include_momentum:
                extras.append("sax_m=0")
            if not self.sax_include_range:
                extras.append("sax_r=0")
            if not self.sax_include_extremes:
                extras.append("sax_e=0")
            if self.sax_include_rsi:
                extras.append(f"sax_rsi={self.sax_rsi_period}")
            if self.sax_use_arrows:
                extras.append("sax_arrows=1")
            if self.sax_use_triangles:
                extras.append("sax_tri=1")
            if self.sax_use_range_symbols:
                extras.append("sax_range_symbols=1")
            if self.sax_use_rsi_symbols:
                extras.append("sax_rsi_symbols=1")
            if self.sax_potential_top_n is not None:
                extras.append(f"sax_top={self.sax_potential_top_n}")
            if self.sax_potential_bucket_count is not None:
                extras.append(f"sax_potb={self.sax_potential_bucket_count}")
            if self.sax_potential_up_only:
                extras.append("sax_potu=1")
            if self.sax_future_shape:
                extras.append(f"sax_fut={self.sax_future_shape_horizon}")
            if self.sax_prospect_buckets:
                extras.append(
                    f"sax_prosp={self.sax_prospect_horizon}:{self.sax_prospect_bucket_count}"
                )
            if self.sax_prospect_time:
                extras.append("sax_prosp_t=1")
        if self.encoding_type == "gauss_returns":
            extras.append(f"gauss_a={self.sax_alphabet_size}")
        if self.encoding_type == "gauss_log_returns":
            extras.append(f"gauss_a={self.sax_alphabet_size}")
            if self.sax_gate_top_n is not None:
                extras.append(f"sax_gate={self.sax_gate_top_n}")
            if self.sax_include_occurrence:
                extras.append("sax_occ=1")
        if self.htf_symbol is not None:
            extras.append(f"htf={self.htf_symbol}")
            extras.append(f"htf_rsi={self.htf_rsi_period}")
        if self.use_cached_content:
            extras.append(f"cache={self.cache_ttl}")
        if self.use_retrieval:
            extras.append(f"ret={self.retrieval_k}")
            if not self.use_retrieval_examples:
                extras.append("ret_ex=0")
        if self.use_sax_retrieval:
            extras.append("sax_ret=1")
        if self.add_patch_summary:
            extras.append(f"patch={self.patch_size}")
        if self.require_summary:
            extras.append("summary=1")
        if self.regime_min_atr_pct is not None:
            extras.append(f"atr>={self.regime_min_atr_pct}")
        if self.label_type == "triple_barrier":
            extras.append(f"h={self.horizon}")
            extras.append(f"bu={self.barrier_up}")
            extras.append(f"bd={self.barrier_down}")
        if self.training_stride is not None:
            extras.append(f"stride={self.training_stride}")
        if self.temperature is not None:
            extras.append(f"temp={self.temperature}")
        if self.top_p is not None:
            extras.append(f"top_p={self.top_p}")
        if self.top_k is not None:
            extras.append(f"top_k={self.top_k}")
        extra_str = "|" + ",".join(extras) if extras else ""
        return (
            f"{self.name}|w={self.window_size}|t={self.training_examples}|"
            f"{self.label_type}|{self.prompt_style}{extra_str}"
        )


# =============================================================================
# LABELS
# =============================================================================


class Direction5(enum.Enum):
    """5-level direction."""

    UP = "UP"  # > +0.3%
    up = "up"  # +0.15% to +0.3%
    same = "same"  # -0.15% to +0.15%
    down = "down"  # -0.3% to -0.15%
    DOWN = "DOWN"  # < -0.3%


class Direction3(enum.Enum):
    """3-level direction."""

    UP = "UP"  # > +0.15%
    SAME = "SAME"  # -0.15% to +0.15%
    DOWN = "DOWN"  # < -0.15%


class Binary(enum.Enum):
    """Binary up/down."""

    UP = "UP"
    DOWN = "DOWN"


def get_label(change_pct: float, label_type: str) -> str:
    """Get label based on type."""
    if label_type == "direction5":
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
    elif label_type == "direction3":
        if change_pct > 0.15:
            return "UP"
        elif change_pct < -0.15:
            return "DOWN"
        else:
            return "SAME"
    elif label_type == "binary":
        return "UP" if change_pct >= 0 else "DOWN"
    elif label_type == "price_pct":
        return f"{change_pct:+.2f}"
    elif label_type == "triple_barrier":
        raise ValueError("triple_barrier labels require future window")
    elif label_type == "hold_bucket":
        raise ValueError("hold_bucket labels require future window")
    elif label_type == "fixed_hold":
        raise ValueError("fixed_hold labels require future window")
    elif label_type == "trailing_barrier":
        raise ValueError("trailing_barrier labels require future window")
    else:
        raise ValueError(f"Unknown label type: {label_type}")


def get_label_description(label_type: str) -> str:
    """Get label description for prompt."""
    if label_type == "direction5":
        return "UP(>+0.3%),up(+0.15-0.3%),same(-0.15-+0.15%),down(-0.3--0.15%),DOWN(<-0.3%)"
    elif label_type == "direction3":
        return "UP(>+0.15%),SAME(-0.15-+0.15%),DOWN(<-0.15%)"
    elif label_type == "binary":
        return "UP(>=0%),DOWN(<0%)"
    elif label_type == "price_pct":
        return "Predict exact % change (e.g., +0.25 or -0.18)"
    elif label_type == "triple_barrier":
        return "UP/DOWN if barrier hit first; same if neither within horizon"
    elif label_type == "hold_bucket":
        return "H4/H6/H8 (hold bars that maximize absolute return)"
    elif label_type == "fixed_hold":
        return "UP/DOWN based on change at fixed horizon"
    elif label_type == "trailing_barrier":
        return "UP if up barrier hit then exit at horizon; DOWN if down barrier first; same if none"
    else:
        raise ValueError(f"Unknown label type: {label_type}")


def is_correct(predicted: str, actual: str, label_type: str) -> bool:
    """Check if prediction is correct."""
    if label_type == "price_pct":
        try:
            pred_val = float(predicted.replace("%", "").replace("+", ""))
            actual_val = float(actual.replace("%", "").replace("+", ""))
            # Correct if same sign and within 0.1%
            return (pred_val * actual_val >= 0) and abs(pred_val - actual_val) < 0.1
        except ValueError:
            return False
    else:
        return predicted == actual


def get_triple_barrier_label(
    hist_prices: np.ndarray,
    future_prices: np.ndarray,
    up_pct: float,
    down_pct: float,
) -> tuple[str, float]:
    """Label based on which barrier is hit first within the horizon."""
    last_price = hist_prices[-1]
    up_level = last_price * (1 + up_pct / 100)
    down_level = last_price * (1 - down_pct / 100)

    for price in future_prices:
        if price >= up_level:
            return "UP", ((price - last_price) / last_price) * 100
        if price <= down_level:
            return "DOWN", ((price - last_price) / last_price) * 100

    change_pct = ((future_prices[-1] - last_price) / last_price) * 100
    return "same", change_pct


def get_hold_bucket_label(
    hist_prices: np.ndarray,
    future_prices: np.ndarray,
    holds: tuple[int, int, int],
) -> tuple[str, float] | None:
    """Label best hold bucket by max absolute return."""
    last_price = hist_prices[-1]
    max_hold = max(holds)
    if len(future_prices) < max_hold:
        return None
    best_hold = None
    best_abs = -1.0
    best_change = 0.0
    for hold in holds:
        price = future_prices[hold - 1]
        change_pct = ((price - last_price) / last_price) * 100
        abs_change = abs(change_pct)
        if abs_change > best_abs or (abs_change == best_abs and (best_hold is None or hold < best_hold)):
            best_abs = abs_change
            best_hold = hold
            best_change = change_pct
    return f"H{best_hold}", best_change


def get_hold_bucket_outcome(
    predicted_label: str,
    hist_prices: np.ndarray,
    future_prices: np.ndarray,
    up_pct: float,
    down_pct: float,
) -> str:
    """Map predicted hold bucket to trade outcome using barriers."""
    if not predicted_label.startswith("H"):
        return "same"
    try:
        hold = int(predicted_label[1:])
    except ValueError:
        return "same"
    if hold <= 0 or len(future_prices) < hold:
        return "same"
    last_price = hist_prices[-1]
    price = future_prices[hold - 1]
    change_pct = ((price - last_price) / last_price) * 100
    if change_pct >= up_pct:
        return "UP"
    if change_pct <= -down_pct:
        return "DOWN"
    return "same"


def get_fixed_hold_label(
    hist_prices: np.ndarray,
    future_prices: np.ndarray,
    horizon: int,
) -> tuple[str, float] | None:
    """Label UP/DOWN based on fixed-horizon return."""
    if len(future_prices) < horizon:
        return None
    last_price = hist_prices[-1]
    price = future_prices[horizon - 1]
    change_pct = ((price - last_price) / last_price) * 100
    label = "UP" if change_pct >= 0 else "DOWN"
    return label, change_pct


def get_trailing_barrier_label(
    hist_prices: np.ndarray,
    future_prices: np.ndarray,
    up_pct: float,
    down_pct: float,
) -> tuple[str, float]:
    """Label with trailing: down barrier exits immediately; up barrier exits at horizon."""
    last_price = hist_prices[-1]
    up_level = last_price * (1 + up_pct / 100)
    down_level = last_price * (1 - down_pct / 100)
    up_hit = False

    for price in future_prices:
        if not up_hit and price <= down_level:
            change_pct = ((price - last_price) / last_price) * 100
            return "DOWN", change_pct
        if not up_hit and price >= up_level:
            up_hit = True

    exit_price = future_prices[-1] if len(future_prices) else last_price
    change_pct = ((exit_price - last_price) / last_price) * 100
    if up_hit and change_pct > 0:
        return "UP", change_pct
    if not up_hit and change_pct == 0:
        return "same", change_pct
    return ("DOWN" if change_pct < 0 else "same"), change_pct


def is_directional(label: str, label_type: str) -> bool:
    """Check if label is directional (not neutral)."""
    if label_type == "direction5":
        return label != "same"
    elif label_type == "direction3":
        return label != "SAME"
    elif label_type == "binary":
        return True
    elif label_type == "price_pct":
        try:
            return abs(float(label.replace("%", "").replace("+", ""))) > 0.15
        except ValueError:
            return False
    elif label_type == "triple_barrier":
        return label != "same"
    elif label_type == "hold_bucket":
        return True
    elif label_type == "fixed_hold":
        return True
    elif label_type == "trailing_barrier":
        return label != "same"
    return False


def to_direction_label(
    label: str,
    label_type: str,
    hist_prices: np.ndarray | None = None,
    future_prices: np.ndarray | None = None,
    config: "ExperimentConfig" | None = None,
) -> str | None:
    """Map label to UP/DOWN/same for directional metrics."""
    if label_type == "direction5":
        if label in {"UP", "up"}:
            return "UP"
        if label in {"DOWN", "down"}:
            return "DOWN"
        if label == "same":
            return "same"
        return None
    if label_type == "direction3":
        if label == "SAME":
            return "same"
        if label in {"UP", "DOWN"}:
            return label
        return None
    if label_type == "binary":
        return label if label in {"UP", "DOWN"} else None
    if label_type in {"triple_barrier", "trailing_barrier"}:
        return label if label in {"UP", "DOWN", "same"} else None
    if label_type == "fixed_hold":
        return label if label in {"UP", "DOWN"} else None
    if label_type == "hold_bucket":
        if hist_prices is None or future_prices is None or config is None:
            return None
        outcome = get_hold_bucket_outcome(
            label,
            hist_prices,
            future_prices,
            config.barrier_up,
            config.barrier_down,
        )
        return outcome if outcome in {"UP", "DOWN", "same"} else None
    return None


def get_valid_labels(label_type: str) -> set[str]:
    if label_type == "hold_bucket":
        return {"H4", "H6", "H8"}
    return {"UP", "up", "same", "down", "DOWN", "SAME"}


# =============================================================================
# ENCODING
# =============================================================================


def _quantize(value: float, scale: float, limit: int = 9) -> int:
    if scale <= 0:
        return 0
    quant = int(round(value / scale))
    return max(-limit, min(limit, quant))


def encode_coordinate(series: np.ndarray, precision: int = 1) -> tuple[str, float]:
    """Encode price series into coordinate format."""
    p90 = np.percentile(np.abs(series), 90)
    scale = p90 if p90 > 0 else 1.0
    scaled = series / scale

    pairs = []
    for idx, val in enumerate(scaled):
        val_int = int(val * (10**precision))
        pairs.append(f"({idx}:{abs(val_int)})")

    return "".join(pairs), scale


def encode_coordinate_with_index(series: np.ndarray, precision: int = 1) -> tuple[str, float]:
    """Encode series with value and explicit index channel."""
    encoded, scale = encode_coordinate(series, precision=precision)
    idx_pairs = [f"{idx}" for idx in range(len(series))]
    return f"{encoded}|IDX:{','.join(idx_pairs)}", scale


def _normalize(series: np.ndarray) -> np.ndarray:
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return series - mean
    return (series - mean) / std


def _paa(series: np.ndarray, segments: int) -> np.ndarray:
    length = len(series)
    if length == segments:
        return np.copy(series)
    result = np.zeros(segments)
    for i in range(segments):
        start = int(i * length / segments)
        end = int((i + 1) * length / segments)
        result[i] = np.mean(series[start:end])
    return result


def _to_sax(series: np.ndarray, segments: int, alphabet_size: int) -> str:
    breakpoints = {
        3: [-0.43, 0.43],
        4: [-0.67, 0.0, 0.67],
        5: [-0.84, -0.25, 0.25, 0.84],
        6: [-0.97, -0.43, 0.0, 0.43, 0.97],
        7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
    }
    if alphabet_size not in breakpoints:
        alphabet_size = 4
    normalized = _normalize(series)
    paa_result = _paa(normalized, segments)
    bp = breakpoints[alphabet_size]
    alphabet = "abcdefg"[:alphabet_size]
    sax_string = ""
    for value in paa_result:
        symbol_idx = 0
        for breakpoint in bp:
            if value > breakpoint:
                symbol_idx += 1
            else:
                break
        if symbol_idx >= len(alphabet):
            symbol_idx = len(alphabet) - 1
        sax_string += alphabet[symbol_idx]
    return sax_string


def encode_sax_enhanced(
    series: np.ndarray,
    segments: int,
    alphabet_size: int,
    include_momentum: bool = True,
    include_range: bool = True,
    include_extremes: bool = True,
    include_rsi: bool = False,
    rsi_period: int = 14,
    use_arrows: bool = False,
    use_triangles: bool = False,
    use_range_symbols: bool = False,
    use_rsi_symbols: bool = False,
) -> tuple[str, float]:
    """Encode series using SAX plus momentum/range indicators."""
    sax_base = _to_sax(series, segments, alphabet_size)

    third = max(1, len(series) // 3)
    first_third = np.mean(series[:third])
    last_third = np.mean(series[-third:])

    momentum = "N"
    if include_momentum:
        if last_third > first_third * 1.002:
            momentum = "U"
        elif last_third < first_third * 0.998:
            momentum = "D"
    if use_arrows:
        momentum = {"U": "↑", "D": "↓", "N": "→"}.get(momentum, momentum)
    if use_triangles:
        momentum = {"U": "▲", "D": "▼", "N": "■"}.get(momentum, momentum)

    alphabet = "abcdefg"[:alphabet_size]
    unique_chars = set(sax_base)
    range_flag = "N"
    if include_range:
        if alphabet[0] in unique_chars and alphabet[-1] in unique_chars:
            range_flag = "F"
        elif len(unique_chars) > alphabet_size * 0.6:
            range_flag = "W"
    if use_range_symbols:
        range_flag = {"F": "▮", "W": "━", "N": "□"}.get(range_flag, range_flag)

    extreme_count = 0
    if include_extremes and sax_base:
        last_char = sax_base[-1]
        if last_char in {alphabet[0], alphabet[-1]}:
            for c in reversed(sax_base):
                if c == last_char:
                    extreme_count += 1
                else:
                    break

    suffix = ""
    if include_momentum:
        suffix += f"|M:{momentum}"
    if include_range:
        suffix += f"|R:{range_flag}"
    if extreme_count > 1:
        suffix += f"|!{extreme_count}"
    if include_rsi:
        rsi_val = _rsi(series, rsi_period)
        bucket = rsi_bucket(rsi_val)
        if use_rsi_symbols:
            bucket = rsi_bucket_symbol(bucket)
        suffix += f"|RSI:{bucket}"

    scale = np.percentile(np.abs(series), 90)
    return f"{sax_base}{suffix}", scale


def _rsi(series: np.ndarray, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(series: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI values for each index using simple averages."""
    if len(series) < period + 1:
        return np.full(len(series), np.nan)
    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi_vals = np.full(len(series), np.nan)
    for i in range(period, len(series)):
        avg_gain = np.mean(gains[i - period : i])
        avg_loss = np.mean(losses[i - period : i])
        if avg_loss == 0:
            rsi_vals[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_vals[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_vals


def encode_sax_rsi(
    series: np.ndarray,
    segments: int,
    alphabet_size: int,
    rsi_period: int = 14,
    use_rsi_symbols: bool = False,
) -> tuple[str, float]:
    """Encode series using SAX plus a single RSI bucket."""
    sax_base = _to_sax(series, segments, alphabet_size)
    rsi_val = _rsi(series, rsi_period)
    if rsi_val < 30:
        rsi_bucket = "L"
    elif rsi_val > 70:
        rsi_bucket = "H"
    else:
        rsi_bucket = "M"
    scale = np.percentile(np.abs(series), 90)
    if use_rsi_symbols:
        rsi_bucket = rsi_bucket_symbol(rsi_bucket)
    return f"{sax_base}|RSI:{rsi_bucket}", scale


def rsi_bucket(rsi_val: float) -> str:
    if np.isnan(rsi_val):
        return "NA"
    if rsi_val < 30:
        return "L"
    if rsi_val > 70:
        return "H"
    return "M"


def rsi_bucket_symbol(bucket: str) -> str:
    return {"L": "▁", "M": "▄", "H": "█"}.get(bucket, bucket)


def encode_signed_coordinate(series: np.ndarray, precision: int = 1) -> tuple[str, float]:
    """Encode signed price deltas to last price with coordinates."""
    last_price = series[-1]
    deltas = (series - last_price) / last_price * 100
    p90 = np.percentile(np.abs(deltas), 90)
    scale = p90 if p90 > 0 else 1.0

    pairs = []
    for idx, val in enumerate(deltas):
        val_int = int(val * (10**precision))
        sign = "-" if val_int < 0 else "+"
        pairs.append(f"({idx}:{sign}{abs(val_int)})")

    return "".join(pairs), scale


def encode_rank(series: np.ndarray) -> tuple[str, float]:
    """Encode per-index rank (0 = lowest)."""
    ranks = series.argsort().argsort()
    pairs = [f"({idx}:r{rank})" for idx, rank in enumerate(ranks)]
    scale = np.percentile(np.abs(series), 90)
    return "".join(pairs), scale


def encode_delta_bins(series: np.ndarray) -> tuple[str, float]:
    """Encode per-index return bins with sign."""
    returns = np.diff(series) / series[:-1] * 100
    bins = [0.05, 0.15, 0.3, 0.6]
    pairs = []
    for idx, ret in enumerate(returns, start=1):
        mag = abs(ret)
        bucket = sum(mag >= b for b in bins)
        sign = "-" if ret < 0 else "+"
        pairs.append(f"({idx}:d{sign}{bucket})")
    scale = np.percentile(np.abs(returns), 90) if len(returns) else 1.0
    return "".join(pairs), scale


def encode_multichannel(prices: np.ndarray, volumes: np.ndarray | None) -> tuple[str, float]:
    """Encode per-index multi-channel tokens."""
    returns = np.diff(prices) / prices[:-1] * 100
    p90 = np.percentile(np.abs(returns), 90) if len(returns) else 1.0
    scale = p90 if p90 > 0 else 1.0

    vol_bins = None
    if volumes is not None:
        v = volumes.copy()
        v_mean = v.mean()
        v_std = v.std() if v.std() > 0 else 1.0
        v_z = (v - v_mean) / v_std
        vol_bins = np.clip(np.round(v_z + 4), 0, 9).astype(int)

    pairs = []
    for idx in range(len(prices)):
        ret = returns[idx - 1] if idx > 0 else 0.0
        q_ret = _quantize(ret, scale)
        sign = "-" if q_ret < 0 else "+"
        v_bin = vol_bins[idx] if vol_bins is not None else 5
        atr_bin = sum(abs(ret) >= b for b in [0.05, 0.15, 0.3])
        pairs.append(f"({idx}:p{sign}{abs(q_ret)}v{v_bin}a{atr_bin})")

    return "".join(pairs), scale


def encode_patch_summary(prices: np.ndarray, patch_size: int) -> str:
    """Append patch-level summaries with absolute patch index."""
    patches = []
    last_price = prices[-1]
    for idx in range(0, len(prices), patch_size):
        patch = prices[idx : idx + patch_size]
        if len(patch) == 0:
            continue
        mean_pct = (patch.mean() - last_price) / last_price * 100
        std_pct = patch.std() / last_price * 100
        trend_pct = (patch[-1] - patch[0]) / patch[0] * 100
        scale = max(0.01, np.percentile(np.abs(prices), 90) / last_price * 100)
        mean_q = _quantize(mean_pct, scale)
        std_q = _quantize(std_pct, scale)
        trend_q = _quantize(trend_pct, scale)
        patches.append(f"[P{idx // patch_size}:{mean_q}/{std_q}/{trend_q}]")
    return "".join(patches)


def compute_summary(prices: np.ndarray, scale: float, encoding_type: str) -> dict[str, int]:
    """Compute summary tokens for min/max delta, trend, volatility."""
    if scale <= 0:
        scale = 1.0

    if encoding_type in {"signed_coordinate", "rank"}:
        last_price = prices[-1]
        deltas = (prices - last_price) / last_price * 100
        min_q = _quantize(deltas.min(), scale)
        max_q = _quantize(deltas.max(), scale)
    else:
        returns = np.diff(prices) / prices[:-1] * 100
        min_q = _quantize(returns.min() if len(returns) else 0.0, scale)
        max_q = _quantize(returns.max() if len(returns) else 0.0, scale)
    trend = 0
    if prices[-1] > prices[0]:
        trend = 1
    elif prices[-1] < prices[0]:
        trend = -1

    returns = np.diff(prices) / prices[:-1] * 100
    vol = np.std(returns) if len(returns) else 0.0
    vol_bin = sum(vol >= b for b in [0.05, 0.15, 0.3])

    return {"min": min_q, "max": max_q, "trend": trend, "vol": vol_bin}


def compute_atr_pct(hist: pd.DataFrame) -> float:
    """Compute ATR% over the provided history."""
    highs = hist["high"].values
    lows = hist["low"].values
    closes = hist["close"].values
    if len(highs) < 2:
        return 0.0
    prev_close = closes[:-1]
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - prev_close))
    tr = np.maximum(tr, np.abs(lows[1:] - prev_close))
    atr = np.mean(tr)
    last_price = closes[-1]
    return (atr / last_price) * 100 if last_price > 0 else 0.0


def build_feature_vector(prices: np.ndarray, volumes: np.ndarray | None) -> np.ndarray:
    """Build retrieval feature vector from prices and volume."""
    returns = np.diff(prices) / prices[:-1] * 100
    p90 = np.percentile(np.abs(returns), 90) if len(returns) else 1.0
    if p90 == 0:
        p90 = 1.0
    ret_vec = returns / p90
    if len(returns):
        slope = np.polyfit(np.arange(len(returns)), returns, 1)[0]
        ret_mean = returns.mean()
        ret_std = returns.std()
    else:
        slope = 0.0
        ret_mean = 0.0
        ret_std = 0.0

    if volumes is None:
        return np.concatenate([ret_vec, [slope, ret_mean, ret_std]])

    v = volumes.astype(float)
    v_mean = v.mean()
    v_std = v.std() if v.std() > 0 else 1.0
    v_z = (v - v_mean) / v_std
    v_z = v_z[1:] if len(v_z) == len(ret_vec) + 1 else v_z[: len(ret_vec)]
    return np.concatenate([ret_vec, v_z, [slope, ret_mean, ret_std]])


def sax_base_from_encoded(encoded: str) -> str:
    """Extract the raw SAX string (before suffix)."""
    return encoded.split("|", 1)[0]


def compute_sax_top_set(
    closes: np.ndarray,
    window_size: int,
    horizon: int,
    paa_segments: int,
    alphabet_size: int,
    top_n: int,
) -> set[str]:
    """Compute top-N SAX bases by average max forward return."""
    scores: dict[str, list[float]] = {}
    last_idx = len(closes) - horizon
    for idx in range(window_size, last_idx):
        window = closes[idx - window_size : idx]
        base = _to_sax(window, paa_segments, alphabet_size)
        last_price = window[-1]
        future = closes[idx : idx + horizon]
        if len(future) == 0 or last_price <= 0:
            continue
        max_ret = (future.max() - last_price) / last_price * 100
        scores.setdefault(base, []).append(max_ret)
    avg = {k: float(np.mean(v)) for k, v in scores.items() if v}
    top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {item[0] for item in top}


def compute_sax_potential_buckets(
    closes: np.ndarray,
    window_size: int,
    horizon: int,
    paa_segments: int,
    alphabet_size: int,
    bucket_count: int,
) -> dict[str, tuple[int, int]]:
    """Bucket SAX bases by average max-up and max-down return potential."""
    if bucket_count < 2:
        return {}

    up_scores: dict[str, list[float]] = {}
    down_scores: dict[str, list[float]] = {}
    last_idx = len(closes) - horizon
    for idx in range(window_size, last_idx):
        window = closes[idx - window_size : idx]
        base = _to_sax(window, paa_segments, alphabet_size)
        last_price = window[-1]
        future = closes[idx : idx + horizon]
        if len(future) == 0 or last_price <= 0:
            continue
        max_ret = (future.max() - last_price) / last_price * 100
        min_ret = (future.min() - last_price) / last_price * 100
        up_scores.setdefault(base, []).append(max_ret)
        down_scores.setdefault(base, []).append(abs(min_ret))

    avg_up = {k: float(np.mean(v)) for k, v in up_scores.items() if v}
    avg_down = {k: float(np.mean(v)) for k, v in down_scores.items() if v}
    if not avg_up or not avg_down:
        return {}

    up_values = np.array(list(avg_up.values()))
    down_values = np.array(list(avg_down.values()))
    up_bins = np.quantile(
        up_values,
        np.linspace(0, 1, bucket_count + 1)[1:-1],
    )
    down_bins = np.quantile(
        down_values,
        np.linspace(0, 1, bucket_count + 1)[1:-1],
    )

    def bucket(value: float, bins: np.ndarray) -> int:
        return int(np.searchsorted(bins, value, side="right"))

    buckets: dict[str, tuple[int, int]] = {}
    for base, up_val in avg_up.items():
        down_val = avg_down.get(base, 0.0)
        buckets[base] = (bucket(up_val, up_bins), bucket(down_val, down_bins))
    return buckets


def encode_returns(series: np.ndarray, precision: int = 2) -> tuple[str, float]:
    """Encode as % returns from previous bar."""
    returns = np.diff(series) / series[:-1] * 100  # % change
    p90 = np.percentile(np.abs(returns), 90)
    scale = p90 if p90 > 0 else 1.0

    pairs = []
    for idx, ret in enumerate(returns):
        sign = "+" if ret >= 0 else "-"
        val = abs(int(ret * (10**precision)))
        pairs.append(f"({idx}:{sign}{val})")

    return "".join(pairs), scale


def encode_gauss_returns(series: np.ndarray, alphabet_size: int = 5) -> tuple[str, float]:
    """Encode returns by Gaussian bins (no PAA), digits 0..N."""
    returns = np.diff(series) / series[:-1] * 100
    if len(returns) == 0:
        return "0", 1.0
    sax_base = _to_sax(returns, len(returns), alphabet_size)
    alphabet = "abcdefg"[:alphabet_size]
    digit_map = {alphabet[i]: str(i) for i in range(len(alphabet))}
    encoded = "".join(digit_map.get(ch, ch) for ch in sax_base)
    scale = np.percentile(np.abs(returns), 90)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return encoded, scale


def encode_gauss_log_returns(series: np.ndarray, alphabet_size: int = 5) -> tuple[str, float]:
    """Encode log-returns by Gaussian bins (no PAA), digits 0..N."""
    safe_series = np.clip(series, 1e-12, None)
    returns = np.diff(np.log(safe_series))
    if len(returns) == 0:
        return "0", 1.0
    sax_base = _to_sax(returns, len(returns), alphabet_size)
    alphabet = "abcdefg"[:alphabet_size]
    digit_map = {alphabet[i]: str(i) for i in range(len(alphabet))}
    encoded = "".join(digit_map.get(ch, ch) for ch in sax_base)
    scale = np.percentile(np.abs(returns), 90)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return encoded, scale


def encode_log_returns(series: np.ndarray, precision: int = 2) -> tuple[str, float]:
    """Encode as log returns from previous bar."""
    returns = np.diff(np.log(series))
    p90 = np.percentile(np.abs(returns), 90)
    scale = p90 if p90 > 0 else 1.0

    pairs = []
    for idx, ret in enumerate(returns):
        sign = "+" if ret >= 0 else "-"
        val = abs(int(ret * (10**precision)))
        pairs.append(f"({idx}:{sign}{val})")

    return "".join(pairs), scale


def encode_sax_log_returns(
    series: np.ndarray,
    segments: int,
    alphabet_size: int,
    include_momentum: bool = True,
    include_range: bool = True,
    include_extremes: bool = True,
    include_rsi: bool = False,
    rsi_period: int = 14,
    use_arrows: bool = False,
    use_triangles: bool = False,
    use_range_symbols: bool = False,
    use_rsi_symbols: bool = False,
) -> tuple[str, float]:
    """Encode log-returns using SAX plus optional flags."""
    returns = np.diff(np.log(series))
    if len(returns) == 0:
        return "a", 1.0
    sax_base = _to_sax(returns, segments, alphabet_size)

    third = max(1, len(returns) // 3)
    first_third = np.mean(returns[:third]) if len(returns) else 0.0
    last_third = np.mean(returns[-third:]) if len(returns) else 0.0

    momentum = "N"
    if include_momentum:
        if last_third > first_third * 1.002:
            momentum = "U"
        elif last_third < first_third * 0.998:
            momentum = "D"
    if use_arrows:
        momentum = {"U": "↑", "D": "↓", "N": "→"}.get(momentum, momentum)
    if use_triangles:
        momentum = {"U": "▲", "D": "▼", "N": "■"}.get(momentum, momentum)

    alphabet = "abcdefg"[:alphabet_size]
    unique_chars = set(sax_base)
    range_flag = "N"
    if include_range:
        if alphabet[0] in unique_chars and alphabet[-1] in unique_chars:
            range_flag = "F"
        elif len(unique_chars) > alphabet_size * 0.6:
            range_flag = "W"
    if use_range_symbols:
        range_flag = {"F": "▮", "W": "━", "N": "□"}.get(range_flag, range_flag)

    extreme_count = 0
    if include_extremes and sax_base:
        last_char = sax_base[-1]
        if last_char in {alphabet[0], alphabet[-1]}:
            for c in reversed(sax_base):
                if c == last_char:
                    extreme_count += 1
                else:
                    break

    suffix = ""
    if include_momentum:
        suffix += f"|M:{momentum}"
    if include_range:
        suffix += f"|R:{range_flag}"
    if extreme_count > 1:
        suffix += f"|!{extreme_count}"
    if include_rsi:
        rsi_val = _rsi(returns, rsi_period)
        bucket = rsi_bucket(rsi_val)
        if use_rsi_symbols:
            bucket = rsi_bucket_symbol(bucket)
        suffix += f"|RSI:{bucket}"

    scale = np.percentile(np.abs(returns), 90) if len(returns) else 1.0
    return f"{sax_base}{suffix}", scale


def encode_future_shape(
    future_prices: np.ndarray,
    last_price: float,
    segments: int,
    alphabet_size: int,
) -> str:
    """Encode future return trajectory as SAX string."""
    if len(future_prices) == 0 or last_price <= 0:
        return "?"
    returns = (future_prices - last_price) / last_price * 100
    if len(returns) < 2:
        return "?"
    return _to_sax(returns, segments, alphabet_size)


def encode_prospect_buckets(
    future_prices: np.ndarray,
    last_price: float,
    bucket_count: int,
) -> str:
    """Encode max-up/max-down magnitudes into buckets."""
    if len(future_prices) == 0 or last_price <= 0 or bucket_count < 2:
        return "U?D?"
    max_ret = (future_prices.max() - last_price) / last_price * 100
    min_ret = (future_prices.min() - last_price) / last_price * 100
    down_mag = abs(min_ret)
    # Use simple fixed buckets based on percentile-like splits (0.5%, 1.5% by default).
    thresholds = np.linspace(0.5, 1.5, bucket_count - 1)
    up_bucket = int(np.searchsorted(thresholds, max_ret, side="right"))
    down_bucket = int(np.searchsorted(thresholds, down_mag, side="right"))
    return f"U{up_bucket}D{down_bucket}"


def encode_prospect_time_buckets(
    future_prices: np.ndarray,
    last_price: float,
    bucket_count: int,
) -> str:
    """Encode time-to-max/time-to-min into buckets."""
    if len(future_prices) == 0 or last_price <= 0 or bucket_count < 2:
        return "TU?TD?"
    max_idx = int(np.argmax(future_prices))
    min_idx = int(np.argmin(future_prices))
    horizon = max(1, len(future_prices) - 1)
    thresholds = np.linspace(0.33, 0.66, bucket_count - 1)

    def bucket(idx: int) -> int:
        frac = idx / horizon
        return int(np.searchsorted(thresholds, frac, side="right"))

    return f"TU{bucket(max_idx)}TD{bucket(min_idx)}"


def encode_with_volume(
    prices: np.ndarray,
    volumes: np.ndarray,
    precision: int = 1,
) -> tuple[str, float]:
    """Encode price with volume indicator."""
    p90_price = np.percentile(np.abs(prices), 90)
    scale = p90_price if p90_price > 0 else 1.0
    scaled_prices = prices / scale

    # Normalize volume to 0-9 scale
    vol_min, vol_max = volumes.min(), volumes.max()
    if vol_max > vol_min:
        vol_norm = (volumes - vol_min) / (vol_max - vol_min) * 9
    else:
        vol_norm = np.full_like(volumes, 5.0)

    pairs = []
    for idx, (price, vol) in enumerate(zip(scaled_prices, vol_norm)):
        p_int = int(price * (10**precision))
        v_int = int(vol)
        pairs.append(f"({idx}:{abs(p_int)}v{v_int})")

    return "".join(pairs), scale


def encode_fft(series: np.ndarray, top_k: int = 5) -> tuple[str, float]:
    """Encode dominant FFT frequencies."""
    fft = np.fft.fft(series)
    freqs = np.fft.fftfreq(len(series))
    magnitudes = np.abs(fft)

    # Get top-k frequencies (skip DC component)
    top_indices = np.argsort(magnitudes[1:])[-top_k:][::-1] + 1

    parts = []
    for idx in top_indices:
        freq = abs(freqs[idx])
        mag = int(magnitudes[idx])
        parts.append(f"(f{freq:.2f}:{mag})")

    scale = np.percentile(np.abs(series), 90)
    return "".join(parts), scale


def encode_natural_language(
    prices: np.ndarray,
    volumes: np.ndarray | None = None,
) -> tuple[str, float]:
    """Encode as natural language description using TA vocabulary."""
    n = len(prices)
    returns = np.diff(prices) / prices[:-1] * 100

    # Overall trend
    start_price, end_price = prices[0], prices[-1]
    total_change = (end_price - start_price) / start_price * 100

    if total_change > 0.5:
        trend = "strong uptrend"
    elif total_change > 0.15:
        trend = "mild uptrend"
    elif total_change < -0.5:
        trend = "strong downtrend"
    elif total_change < -0.15:
        trend = "mild downtrend"
    else:
        trend = "sideways/ranging"

    # Volatility
    volatility = np.std(returns)
    if volatility > 0.3:
        vol_desc = "high volatility"
    elif volatility > 0.15:
        vol_desc = "moderate volatility"
    else:
        vol_desc = "low volatility"

    # Recent momentum (last 5 bars)
    recent = returns[-5:] if len(returns) >= 5 else returns
    recent_avg = np.mean(recent)
    if recent_avg > 0.1:
        momentum = "bullish momentum building"
    elif recent_avg < -0.1:
        momentum = "bearish pressure"
    else:
        momentum = "neutral momentum"

    # Pattern detection
    patterns = []

    # Higher highs / lower lows
    highs = [
        prices[i]
        for i in range(1, n - 1)
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]
    ]
    lows = [
        prices[i]
        for i in range(1, n - 1)
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]
    ]

    if len(highs) >= 2 and highs[-1] > highs[-2]:
        patterns.append("higher highs")
    if len(highs) >= 2 and highs[-1] < highs[-2]:
        patterns.append("lower highs")
    if len(lows) >= 2 and lows[-1] > lows[-2]:
        patterns.append("higher lows")
    if len(lows) >= 2 and lows[-1] < lows[-2]:
        patterns.append("lower lows")

    # Support/resistance tests
    recent_high = max(prices[-5:])
    recent_low = min(prices[-5:])
    if prices[-1] >= recent_high * 0.998:
        patterns.append("testing resistance")
    if prices[-1] <= recent_low * 1.002:
        patterns.append("testing support")

    # Reversal signals
    if len(returns) >= 3:
        if returns[-3] < -0.1 and returns[-2] < -0.1 and returns[-1] > 0.1:
            patterns.append("potential bullish reversal")
        if returns[-3] > 0.1 and returns[-2] > 0.1 and returns[-1] < -0.1:
            patterns.append("potential bearish reversal")

    # Volume analysis
    vol_desc_extra = ""
    if volumes is not None and len(volumes) >= 5:
        recent_vol = np.mean(volumes[-3:])
        older_vol = np.mean(volumes[:-3])
        if recent_vol > older_vol * 1.5:
            vol_desc_extra = " Volume surging."
        elif recent_vol < older_vol * 0.5:
            vol_desc_extra = " Volume declining."

    # Build description
    pattern_str = ", ".join(patterns) if patterns else "no clear pattern"

    desc = (
        f"{trend.capitalize()}. {vol_desc.capitalize()}. {momentum.capitalize()}. "
        f"Patterns: {pattern_str}.{vol_desc_extra}"
    )

    scale = np.percentile(np.abs(prices), 90)
    return desc, scale


# =============================================================================
# DATA & TRAINING
# =============================================================================


def load_data(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Load OHLCV data."""
    path = DATA_PATH / symbol
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_htf_rsi_lookup(htf_df: pd.DataFrame, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Build lookup arrays for higher-timeframe RSI values."""
    ts = pd.to_datetime(htf_df["timestamp"]).values.astype("datetime64[ns]")
    closes = htf_df["close"].values
    rsi_vals = rsi_series(closes, period)
    return ts, rsi_vals


def get_htf_rsi_bucket(
    htf_ts: np.ndarray,
    htf_rsi_vals: np.ndarray,
    ts: pd.Timestamp,
    use_symbols: bool = False,
) -> str:
    """Lookup latest HTF RSI bucket for a timestamp."""
    idx = np.searchsorted(htf_ts, ts.to_datetime64(), side="right") - 1
    if idx < 0:
        return "NA"
    bucket = rsi_bucket(htf_rsi_vals[idx])
    if use_symbols:
        return rsi_bucket_symbol(bucket)
    return bucket


def do_encode(
    prices: np.ndarray,
    volumes: np.ndarray | None,
    encoding_type: str,
    add_patch_summary: bool,
    patch_size: int,
    sax_paa_segments: int,
    sax_alphabet_size: int,
    sax_include_momentum: bool,
    sax_include_range: bool,
    sax_include_extremes: bool,
    sax_include_rsi: bool,
    sax_rsi_period: int,
    sax_use_arrows: bool,
    sax_use_triangles: bool,
    sax_use_range_symbols: bool,
    sax_use_rsi_symbols: bool,
    add_index_channel: bool = False,
) -> tuple[str, float]:
    """Encode based on type."""
    if encoding_type == "returns":
        return encode_returns(prices)
    elif encoding_type == "gauss_returns":
        return encode_gauss_returns(prices, sax_alphabet_size)
    elif encoding_type == "gauss_log_returns":
        return encode_gauss_log_returns(prices, sax_alphabet_size)
    elif encoding_type == "log_returns":
        return encode_log_returns(prices)
    elif encoding_type == "volume" and volumes is not None:
        return encode_with_volume(prices, volumes)
    elif encoding_type == "fft":
        return encode_fft(prices)
    elif encoding_type == "natural":
        return encode_natural_language(prices, volumes)
    elif encoding_type == "signed_coordinate":
        encoded, scale = encode_signed_coordinate(prices)
    elif encoding_type == "rank":
        encoded, scale = encode_rank(prices)
    elif encoding_type == "delta_bins":
        encoded, scale = encode_delta_bins(prices)
    elif encoding_type == "multichannel":
        encoded, scale = encode_multichannel(prices, volumes)
    elif encoding_type == "sax":
        encoded, scale = encode_sax_enhanced(
            prices,
            sax_paa_segments,
            sax_alphabet_size,
            sax_include_momentum,
            sax_include_range,
            sax_include_extremes,
            sax_include_rsi,
            sax_rsi_period,
            sax_use_arrows,
            sax_use_triangles,
            sax_use_range_symbols,
            sax_use_rsi_symbols,
        )
    elif encoding_type == "sax_log_returns":
        encoded, scale = encode_sax_log_returns(
            prices,
            sax_paa_segments,
            sax_alphabet_size,
            sax_include_momentum,
            sax_include_range,
            sax_include_extremes,
            sax_include_rsi,
            sax_rsi_period,
            sax_use_arrows,
            sax_use_triangles,
            sax_use_range_symbols,
            sax_use_rsi_symbols,
        )
    elif encoding_type == "sax_rsi":
        encoded, scale = encode_sax_rsi(
            prices,
            sax_paa_segments,
            sax_alphabet_size,
            sax_rsi_period,
            sax_use_rsi_symbols,
        )
    else:  # coordinate (default)
        if add_index_channel and encoding_type in {"coordinate", "signed_coordinate"}:
            encoded, scale = encode_coordinate_with_index(prices)
        else:
            encoded, scale = encode_coordinate(prices)

    if add_patch_summary:
        encoded = f"{encoded}{encode_patch_summary(prices, patch_size)}"
    return encoded, scale


def generate_examples(
    df: pd.DataFrame,
    config: ExperimentConfig,
    htf_ts: np.ndarray | None = None,
    htf_rsi_vals: np.ndarray | None = None,
    sax_counts: dict[str, int] | None = None,
    sax_potential_buckets: dict[str, tuple[int, int]] | None = None,
) -> list[dict]:
    """Generate training examples."""
    if config.training_examples == 0:
        return []

    examples = []
    if config.training_stride is not None:
        step_size = max(1, config.training_stride)
    else:
        step_size = max(1, len(df) // config.training_examples)

    for i in range(0, len(df) - config.window_size - 1, step_size):
        if len(examples) >= config.training_examples:
            break
        if i + config.window_size + 1 >= len(df):
            continue

        hist = df.iloc[i : i + config.window_size]
        future_idx = i + config.window_size
        future = df.iloc[future_idx]
        horizon_end = min(len(df), future_idx + config.horizon)
        future_window = df.iloc[future_idx:horizon_end]

        hist_prices = hist["close"].values
        hist_volumes = hist["volume"].values if "volume" in hist.columns else None
        future_price = future["close"]
        last_price = hist_prices[-1]
        change_pct = ((future_price - last_price) / last_price) * 100

        encoded, scale = do_encode(
            hist_prices,
            hist_volumes,
            config.encoding_type,
            config.add_patch_summary,
            config.patch_size,
            config.sax_paa_segments,
            config.sax_alphabet_size,
            config.sax_include_momentum,
            config.sax_include_range,
            config.sax_include_extremes,
            config.sax_include_rsi,
            config.sax_rsi_period,
            config.sax_use_arrows,
            config.sax_use_triangles,
            config.sax_use_range_symbols,
            config.sax_use_rsi_symbols,
            config.add_index_channel,
        )
        if htf_ts is not None and htf_rsi_vals is not None:
            htf_bucket = get_htf_rsi_bucket(
                htf_ts,
                htf_rsi_vals,
                hist["timestamp"].iloc[-1],
                config.sax_use_rsi_symbols,
            )
            encoded = f"{encoded}|HTF_RSI:{htf_bucket}"
        if (
            config.encoding_type == "sax"
            and sax_counts is not None
            and config.sax_include_occurrence
        ):
            base = sax_base_from_encoded(encoded)
            occ = sax_counts.get(base, 0)
            if occ < 10:
                occ_bucket = "L"
            elif occ < 30:
                occ_bucket = "M"
            else:
                occ_bucket = "H"
            encoded = f"{encoded}|OCC:{occ_bucket}"
        if config.encoding_type == "sax" and sax_potential_buckets is not None:
            base = sax_base_from_encoded(encoded)
            buckets = sax_potential_buckets.get(base)
            if buckets is None:
                token = "U?" if config.sax_potential_up_only else "U?D?"
            elif config.sax_potential_up_only:
                token = f"U{buckets[0]}"
            else:
                token = f"U{buckets[0]}D{buckets[1]}"
            encoded = f"{encoded}|POT:{token}"
        if config.encoding_type == "sax" and config.sax_future_shape:
            fut_prices = future_window["close"].values[: config.sax_future_shape_horizon]
            fut_shape = encode_future_shape(
                fut_prices,
                last_price,
                config.sax_future_shape_segments,
                config.sax_alphabet_size,
            )
            encoded = f"{encoded}|FUT:{fut_shape}"
        if config.encoding_type == "sax" and config.sax_prospect_buckets:
            fut_prices = future_window["close"].values[: config.sax_prospect_horizon]
            prospect = encode_prospect_buckets(
                fut_prices,
                last_price,
                config.sax_prospect_bucket_count,
            )
            encoded = f"{encoded}|PROS:{prospect}"
        if config.encoding_type == "sax" and config.sax_prospect_time:
            fut_prices = future_window["close"].values[: config.sax_prospect_horizon]
            prospect = encode_prospect_time_buckets(
                fut_prices,
                last_price,
                config.sax_prospect_bucket_count,
            )
            encoded = f"{encoded}|PT:{prospect}"
        if config.label_type == "triple_barrier":
            label, change_pct = get_triple_barrier_label(
                hist_prices,
                future_window["close"].values,
                config.barrier_up,
                config.barrier_down,
            )
        elif config.label_type == "hold_bucket":
            hold_label = get_hold_bucket_label(
                hist_prices,
                future_window["close"].values,
                config.hold_buckets,
            )
            if hold_label is None:
                continue
            label, change_pct = hold_label
        elif config.label_type == "fixed_hold":
            fixed_label = get_fixed_hold_label(
                hist_prices,
                future_window["close"].values,
                config.horizon,
            )
            if fixed_label is None:
                continue
            label, change_pct = fixed_label
        elif config.label_type == "trailing_barrier":
            label, change_pct = get_trailing_barrier_label(
                hist_prices,
                future_window["close"].values,
                config.barrier_up,
                config.barrier_down,
            )
        else:
            label = get_label(change_pct, config.label_type)

        features = build_feature_vector(hist_prices, hist_volumes)

        examples.append(
            {
                "encoded": encoded,
                "scale": scale,
                "last_price": last_price,
                "label": label,
                "change_pct": change_pct,
                "features": features,
            }
        )

    return examples


def build_system_prompt(config: ExperimentConfig) -> str:
    """Build system prompt based on config and style."""
    summary_block = ""
    if config.require_summary:
        summary_block = (
            "\n## SUMMARY CHECK\n"
            "First output: SUMMARY|min:{min}|max:{max}|trend:{trend}|vol:{vol}\n"
            "min/max are quantized using the SCALE from input (-9..9).\n"
            "trend is -1/0/1 based on first vs last value.\n"
            "vol is bin 0-3 based on return volatility.\n"
            "Then output the prediction line.\n"
        )

    if config.prompt_style == "minimal":
        return f"""Predict next 15min price direction from encoded candles.
Labels: {get_label_description(config.label_type)}
Format: PREDICTION|CONFIDENCE (e.g., UP|HIGH)
{summary_block}
"""
    elif config.prompt_style == "strict":
        return f"""TASK: Predict next move.
Labels: {get_label_description(config.label_type)}
Format: PREDICTION|CONFIDENCE (single line only)
{summary_block}
"""
    elif config.prompt_style == "multi_vote":
        return f"""TASK: Predict next move.
Labels: {get_label_description(config.label_type)}
Give 3 independent guesses on separate lines.
Format each line: PREDICTION|CONFIDENCE
The 3 guesses should not be identical unless very confident.
{summary_block}
"""
    elif config.prompt_style == "multi_vote5":
        return f"""TASK: Predict next move.
Labels: {get_label_description(config.label_type)}
Give 5 independent guesses on separate lines.
Format each line: PREDICTION|CONFIDENCE
The 5 guesses should not be identical unless very confident.
{summary_block}
"""

    elif config.prompt_style == "cot":
        return f"""You are an expert crypto trader analyzing
{config.window_size}-bar price patterns.

## TASK
Predict next 15-minute price direction using chain-of-thought reasoning.

## DATA
- Encoded candles: (position:scaled_value) pairs
- Scale: 90th percentile normalization factor
- Labels: {get_label_description(config.label_type)}

## ANALYSIS STEPS (think through each)
1. SPIKES: Count values of 10. More spikes = more volatility.
2. PATTERN: Where are spikes? Early (0-3) = leading indicator. Late (8+) = trailing.
3. TREND: Are values increasing/decreasing across positions?
4. SCALE: High scale (>3000) = volatile period. Low (<1500) = quiet.
5. SIMILAR: Which training examples match this pattern?

## RESPONSE
First analyze, then give: PREDICTION|CONFIDENCE
{summary_block}
"""

    elif config.prompt_style == "multi_answer":
        return """You are a cryptocurrency trading analyst.
For each pattern, provide THREE possible predictions ranked by probability.

## TASK
Predict the next 15-minute price movement. Give 3 answers with probability estimates.

## DATA FORMAT
- TIME SERIES: (coordinate:scaled_value) pairs, {config.window_size} timesteps
- SCALE: 90th percentile normalization
- Labels: {get_label_description(config.label_type)}

## RESPONSE FORMAT (exactly 3 lines)
1. PREDICTION|PROBABILITY% (most likely)
2. PREDICTION|PROBABILITY% (second)
3. PREDICTION|PROBABILITY% (third)

Example:
1. UP|45%
2. same|35%
3. down|20%

Probabilities must sum to 100%. Pick the BEST answer, not the safest.
{summary_block}
"""

    elif config.prompt_style == "trader":
        return """You are a professional crypto day trader analyzing ETH/USD
on the 15-minute chart.

## YOUR EDGE
You've studied thousands of patterns. You know:
- Higher highs + higher lows = continuation likely
- Lower highs + testing support = breakdown risk
- Bullish momentum after consolidation = breakout setup
- Volume surge confirms moves, declining volume = fake out

## TASK
Read the market description and predict the next 15-minute candle.

## LABELS
{get_label_description(config.label_type)}

## RESPONSE
Give your read: PREDICTION|CONFIDENCE
Be decisive. Trust your pattern recognition.
{summary_block}
"""

    elif config.prompt_style == "contrarian":
        return f"""You are a contrarian trader who profits by fading the crowd.

## YOUR EDGE
Most traders lose money. You win by:
- Fading euphoria (everyone bullish = sell signal)
- Buying panic (everyone bearish = buy signal)
- Spotting exhaustion in trends
- Taking the OTHER side of obvious patterns

## TASK
A move IS happening. Predict: UP or DOWN?
You MUST pick a direction. No fence-sitting.

## RESPONSE FORMAT
DIRECTION|CONFIDENCE (UP or DOWN only)
{summary_block}
"""

    elif config.prompt_style == "sentiment":
        return f"""You are analyzing crowd sentiment in crypto markets.

## DATA FORMAT
Each number represents trader sentiment (0-10 scale):
- 10 = extreme greed/euphoria
- 5 = neutral
- 0 = extreme fear/panic

## MARKET PSYCHOLOGY
- Euphoria (9-10) often precedes drops
- Panic (0-2) often precedes bounces
- Divergence between sentiment and price = reversal signal

## TASK
Predict next sentiment shift: {get_label_description(config.label_type)}

## RESPONSE
PREDICTION|CONFIDENCE
{summary_block}
"""

    elif config.prompt_style == "fewshot":
        return f"""You are a pattern matcher. Learn from these examples.

## RULES (discovered from data):
1. Three+ rising values ending at 10 → likely DOWN (exhaustion)
2. Three+ falling values ending at 9 → likely UP (bounce)
3. Oscillating 9-10-9-10 → likely same direction as last move
4. Sudden spike to 10 after flat 9s → likely DOWN (spike reversal)
5. Values stuck at 9 for 5+ positions → breakout coming (direction unclear)

## KEY INSIGHT
The number 10 often marks a LOCAL EXTREME. After 10, expect mean reversion.

## RESPONSE FORMAT
{get_label_description(config.label_type)}
Answer: PREDICTION|CONFIDENCE
{summary_block}
"""

    elif config.prompt_style == "zeroshot":
        return f"""CRYPTO 15-MIN DIRECTION PREDICTION

You see coordinate-encoded price data: (index:normalized_value) pairs.
Values are 0-10 scale where 10 = local high, 9 = typical.

MAKE A CALL: Will price go UP or DOWN in next 15 minutes?

DECISION FRAMEWORK:
- Recent trend continuation is likely (momentum)
- BUT extremes (multiple 10s) often reverse
- Last few positions matter most

OUTPUT: Just the direction: UP or DOWN
{summary_block}
"""

    else:  # detailed (default)
        return f"""You are a cryptocurrency trading analyst specializing in pattern recognition.

## TASK
Predict the next 15-minute price movement from coordinate-encoded candle patterns.

## DATA FORMAT
- TIME SERIES: (coordinate:scaled_value) pairs, {config.window_size} timesteps
- SCALE: 90th percentile normalization
- PRICE: Current price

## LABELS
{get_label_description(config.label_type)}

## PATTERN RULES
1. Count "10" values (spikes) - 0-1 spikes: neutral, 2+ consecutive: strong signal
2. Early positions (0-3) = strong, mid (4-7) = medium, late (8+) = weak
3. Scale > 4000: high volatility (stronger signals), < 1500: low volatility (neutral bias)
4. Clustered spikes = momentum, scattered = reversal potential

## RESPONSE FORMAT
PREDICTION|CONFIDENCE
Example: UP|HIGH or DOWN|LOW or same|MEDIUM
{summary_block}
"""


def format_training_context(examples: list[dict], config: ExperimentConfig) -> str:
    """Format training examples."""
    prompt = build_system_prompt(config)

    # Zero-shot: no training examples
    if config.prompt_style == "zeroshot" or len(examples) == 0:
        return prompt

    lines = [prompt, "", "## TRAINING EXAMPLES", ""]

    for i, ex in enumerate(examples):
        line = (
            f"#{i + 1} "
            f"{ex['encoded']}|{ex['scale']:.0f}|{ex['last_price']:.0f}→{ex['label']}"
        )
        lines.append(line)

    return "\n".join(lines)


def select_retrieval_examples(
    examples: list[dict],
    query_features: np.ndarray,
    k: int,
) -> list[dict]:
    """Select nearest neighbor examples for retrieval-ICL."""
    if not examples:
        return []
    feats = np.stack([ex["features"] for ex in examples])
    q = query_features
    if q.shape[0] != feats.shape[1]:
        min_len = min(q.shape[0], feats.shape[1])
        feats = feats[:, :min_len]
        q = q[:min_len]
    dists = np.linalg.norm(feats - q, axis=1)
    idx = np.argsort(dists)[:k]
    return [examples[i] for i in idx]


def select_retrieval_examples_sax(
    examples: list[dict],
    query_encoded: str,
    k: int,
) -> list[dict]:
    """Select nearest neighbor examples by SAX base Hamming distance."""
    if not examples:
        return []
    query_base = sax_base_from_encoded(query_encoded)
    q_len = len(query_base)
    dists = []
    for idx, ex in enumerate(examples):
        base = sax_base_from_encoded(ex["encoded"])
        min_len = min(q_len, len(base))
        dist = abs(q_len - len(base))
        if min_len:
            dist += sum(1 for i in range(min_len) if base[i] != query_base[i])
        dists.append((dist, idx))
    dists.sort(key=lambda x: x[0])
    return [examples[i] for _, i in dists[:k]]


# =============================================================================
# GEMINI PREDICTOR
# =============================================================================


class GeminiPredictor:
    """Gemini predictor with configurable options."""

    def __init__(self, context: str, config: ExperimentConfig):
        from google import genai

        self.client = genai.Client()
        self.model = config.model
        self.context = context
        self.config = config
        self.chat = None
        self.cached_content_name = None
        self.cache_key = None
        if config.use_cached_content:
            digest = hashlib.sha1(context.encode("utf-8")).hexdigest()
            self.cache_key = f"{config.model}:{digest}"

    def start(self) -> None:
        """Start predictor."""
        if self.config.use_chat:
            self.chat = self.client.chats.create(model=self.model)
            self.chat.send_message(self.context)
            print("[Gemini] Chat session started")
        elif self.config.use_cached_content:
            from google.genai import types

            display_name = f"llm_predict:{self.cache_key}"
            try:
                for cache in self.client.caches.list():
                    if getattr(cache, "display_name", None) == display_name:
                        self.cached_content_name = cache.name
                        print(f"[Gemini] Using cached_content_name={cache.name}")
                        return
            except Exception as e:
                print(f"[Gemini] Cache list failed, creating new cache: {e}")

            try:
                cache = self.client.caches.create(
                    model=self.model,
                    config=types.CreateCachedContentConfig(
                        display_name=display_name,
                        contents=[self.context],
                        ttl=self.config.cache_ttl,
                    ),
                )
                self.cached_content_name = cache.name
                print(f"[Gemini] Cached content: {cache.name}")
            except Exception as e:
                print(f"[Gemini] Cache create failed, using implicit caching: {e}")
        else:
            print("[Gemini] Single-request mode (implicit caching)")

    def predict(
        self,
        encoded: str,
        scale: float,
        price: float,
        context_override: str | None = None,
    ) -> tuple[str, str, dict | None]:
        """Make prediction. Returns (label, confidence, summary)."""
        from google.genai import types

        query = f"{encoded}|{scale:.0f}|{price:.0f}"
        max_tokens = 150 if self.config.prompt_style == "multi_answer" else 50

        for attempt in range(3):
            try:
                if self.config.use_chat and context_override is None:
                    response = self.chat.send_message(
                        f"## TEST\n{query}",
                        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
                    )
                else:
                    context = context_override if context_override is not None else self.context
                    prompt = f"{context}\n\n## TEST\n{query}"
                    gen_config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                    if self.cached_content_name and context_override is None:
                        gen_config.cached_content = self.cached_content_name
                        prompt = f"## TEST\n{query}"
                    if self.config.temperature is not None:
                        gen_config.temperature = self.config.temperature
                    if self.config.top_p is not None:
                        gen_config.top_p = self.config.top_p
                    if self.config.top_k is not None:
                        gen_config.top_k = self.config.top_k
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=gen_config,
                    )

                if response.usage_metadata:
                    cached_tokens = getattr(
                        response.usage_metadata,
                        "cached_content_token_count",
                        0,
                    )
                    if cached_tokens:
                        print(f"[Gemini] cached_content_token_count={cached_tokens}")
                    prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                    total_tokens = getattr(response.usage_metadata, "total_token_count", 0)
                    if prompt_tokens or total_tokens:
                        print(f"[Gemini] prompt_tokens={prompt_tokens} total_tokens={total_tokens}")
                text = response.text.strip() if response.text else ""
                return self._parse(text)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"[Rate limit] Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return "same", "LOW", None  # Fallback

    def _parse(self, text: str) -> tuple[str, str, dict | None]:
        """Parse PREDICTION|CONFIDENCE or multi_answer response."""
        valid_labels = get_valid_labels(self.config.label_type)
        summary = None

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("SUMMARY"):
                parts = line.replace("SUMMARY", "").strip(" :|").split("|")
                summary = {}
                for part in parts:
                    if ":" in part:
                        key, val = part.split(":", 1)
                        key = key.strip()
                        try:
                            summary[key] = int(val.strip())
                        except ValueError:
                            pass

        # For multi_answer, take first line's prediction
        if self.config.prompt_style == "multi_answer":
            for line in text.split("\n"):
                # Match "1. UP|45%" or "UP|45%"
                line = line.strip()
                if line.startswith("1."):
                    line = line[2:].strip()
                if "|" in line:
                    parts = line.split("|")
                    label = parts[0].strip()
                    if label in valid_labels:
                        conf = parts[1].strip().replace("%", "")
                        return label, conf, summary
            return "same", "33", summary

        if self.config.prompt_style == "multi_vote":
            labels = []
            for line in text.split("\n"):
                if "|" in line:
                    label = line.strip().split("|", 1)[0].strip()
                    if label in valid_labels:
                        labels.append(label)
            if labels:
                counts = {label: labels.count(label) for label in labels}
                best = max(counts, key=counts.get)
                return best, f"{counts[best]}/{len(labels)}", summary
            return "same", "0/0", summary

        # Standard parsing
        for line in text.split("\n"):
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    label = parts[0].strip()
                    if label in valid_labels:
                        return label, parts[1].strip(), summary
        return "same", "LOW", summary

    def cleanup(self) -> None:
        """Cleanup."""
        self.chat = None


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


@dataclass
class FoldResult:
    """Results from one fold."""

    total: int
    correct: int
    directional_total: int
    directional_correct: int
    invalid_summary: int = 0
    dir_confusion: dict[str, dict[str, int]] | None = None
    dir_return_sum: float = 0.0
    dir_return_count: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0

    @property
    def directional_accuracy(self) -> float:
        return (
            self.directional_correct / self.directional_total if self.directional_total > 0 else 0
        )


def run_fold(
    predictor: GeminiPredictor,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    fold_num: int,
    training_examples: list[dict],
    htf_ts: np.ndarray | None = None,
    htf_rsi_vals: np.ndarray | None = None,
    sax_gate: set[str] | None = None,
    sax_potential_buckets: dict[str, tuple[int, int]] | None = None,
) -> FoldResult:
    """Run one fold of testing."""
    print(f"\n--- Fold {fold_num + 1}/{config.num_folds} ---")

    correct = 0
    total = 0
    dir_correct = 0
    dir_total = 0
    invalid_summary = 0
    dir_labels = ["UP", "DOWN", "same"]
    dir_confusion = {a: {p: 0 for p in dir_labels} for a in dir_labels}
    dir_returns: list[float] = []

    # Sample evenly across test data
    step = max(1, len(test_df) // config.num_tests)

    for i in range(config.window_size, len(test_df) - 1, step):
        if total >= config.num_tests:
            break
        if i + 1 >= len(test_df):
            continue

        hist = test_df.iloc[i - config.window_size : i]
        if config.regime_min_atr_pct is not None:
            atr_pct = compute_atr_pct(hist.tail(config.atr_lookback + 1))
            if atr_pct < config.regime_min_atr_pct:
                continue
        future = test_df.iloc[i]
        horizon_end = min(len(test_df), i + config.horizon)
        future_window = test_df.iloc[i:horizon_end]

        hist_prices = hist["close"].values
        hist_volumes = hist["volume"].values if "volume" in hist.columns else None
        future_price = future["close"]
        last_price = hist_prices[-1]
        change_pct = ((future_price - last_price) / last_price) * 100

        encoded, scale = do_encode(
            hist_prices,
            hist_volumes,
            config.encoding_type,
            config.add_patch_summary,
            config.patch_size,
            config.sax_paa_segments,
            config.sax_alphabet_size,
            config.sax_include_momentum,
            config.sax_include_range,
            config.sax_include_extremes,
            config.sax_include_rsi,
            config.sax_rsi_period,
            config.sax_use_arrows,
            config.sax_use_triangles,
            config.sax_use_range_symbols,
            config.sax_use_rsi_symbols,
            config.add_index_channel,
        )
        if htf_ts is not None and htf_rsi_vals is not None:
            htf_bucket = get_htf_rsi_bucket(
                htf_ts,
                htf_rsi_vals,
                hist["timestamp"].iloc[-1],
                config.sax_use_rsi_symbols,
            )
            encoded = f"{encoded}|HTF_RSI:{htf_bucket}"
        if config.label_type == "triple_barrier":
            actual, change_pct = get_triple_barrier_label(
                hist_prices,
                future_window["close"].values,
                config.barrier_up,
                config.barrier_down,
            )
        elif config.label_type == "hold_bucket":
            hold_label = get_hold_bucket_label(
                hist_prices,
                future_window["close"].values,
                config.hold_buckets,
            )
            if hold_label is None:
                continue
            actual, change_pct = hold_label
        elif config.label_type == "fixed_hold":
            fixed_label = get_fixed_hold_label(
                hist_prices,
                future_window["close"].values,
                config.horizon,
            )
            if fixed_label is None:
                continue
            actual, change_pct = fixed_label
        elif config.label_type == "trailing_barrier":
            actual, change_pct = get_trailing_barrier_label(
                hist_prices,
                future_window["close"].values,
                config.barrier_up,
                config.barrier_down,
            )
        else:
            actual = get_label(change_pct, config.label_type)

        if config.encoding_type == "sax" and sax_gate is not None:
            base = sax_base_from_encoded(encoded)
            if base not in sax_gate:
                pred = "same"
                conf = "LOW"
                summary = None
                is_right = is_correct(pred, actual, config.label_type)
                if is_right:
                    correct += 1
                total += 1
                if is_directional(actual, config.label_type):
                    dir_total += 1
                    if is_right:
                        dir_correct += 1
                status = "+" if is_right else "x"
                print(f"[{total}] {status} P:{pred:5} A:{actual:5} Δ:{change_pct:+.2f}%")
                time.sleep(0.5)
                continue

        if config.encoding_type == "sax" and sax_potential_buckets is not None:
            base = sax_base_from_encoded(encoded)
            buckets = sax_potential_buckets.get(base)
            if buckets is None:
                token = "U?" if config.sax_potential_up_only else "U?D?"
            elif config.sax_potential_up_only:
                token = f"U{buckets[0]}"
            else:
                token = f"U{buckets[0]}D{buckets[1]}"
            encoded = f"{encoded}|POT:{token}"
        if config.encoding_type == "sax" and config.sax_future_shape:
            encoded = f"{encoded}|FUT:?"
        if config.encoding_type == "sax" and config.sax_prospect_buckets:
            encoded = f"{encoded}|PROS:U?D?"
        if config.encoding_type == "sax" and config.sax_prospect_time:
            encoded = f"{encoded}|PT:TU?TD?"
        if config.encoding_type == "sax" and config.sax_potential_top_n is not None:
            encoded = f"{encoded}|TOP:?"

        context_override = None
        if config.use_retrieval:
            if not config.use_retrieval_examples:
                neighbors = []
            elif config.use_sax_retrieval and config.encoding_type.startswith("sax"):
                neighbors = select_retrieval_examples_sax(
                    training_examples,
                    encoded,
                    config.retrieval_k,
                )
            else:
                query_features = build_feature_vector(hist_prices, hist_volumes)
                neighbors = select_retrieval_examples(
                    training_examples,
                    query_features,
                    config.retrieval_k,
                )
            context_override = format_training_context(neighbors, config)

        pred, conf, summary = predictor.predict(
            encoded,
            scale,
            last_price,
            context_override=context_override,
        )
        if config.require_summary:
            expected = compute_summary(hist_prices, scale, config.encoding_type)
            if summary != expected:
                invalid_summary += 1
                pred = "same"

        is_right = is_correct(pred, actual, config.label_type)
        if is_right:
            correct += 1
        total += 1

        if config.label_type == "hold_bucket":
            dir_total += 1
            outcome = get_hold_bucket_outcome(
                pred,
                hist_prices,
                future_window["close"].values,
                config.barrier_up,
                config.barrier_down,
            )
            if outcome == "UP":
                dir_correct += 1
        elif is_directional(actual, config.label_type):
            dir_total += 1
            if is_right:
                dir_correct += 1

        actual_dir = to_direction_label(
            actual,
            config.label_type,
            hist_prices,
            future_window["close"].values,
            config,
        )
        pred_dir = to_direction_label(
            pred,
            config.label_type,
            hist_prices,
            future_window["close"].values,
            config,
        )
        if actual_dir is not None and pred_dir is not None:
            dir_confusion[actual_dir][pred_dir] += 1
            if pred_dir == "UP":
                dir_returns.append(change_pct)
            elif pred_dir == "DOWN":
                dir_returns.append(-change_pct)
            else:
                dir_returns.append(0.0)

        status = "+" if is_right else "x"
        print(f"[{total}] {status} P:{pred:5} A:{actual:5} Δ:{change_pct:+.2f}%")

        time.sleep(0.5)  # Rate limit

    if any(sum(v.values()) for v in dir_confusion.values()):
        for label in dir_labels:
            pred_total = sum(dir_confusion[a][label] for a in dir_labels)
            actual_total = sum(dir_confusion[label].values())
            tp = dir_confusion[label][label]
            precision = tp / pred_total if pred_total else 0.0
            recall = tp / actual_total if actual_total else 0.0
            print(f"{label} precision: {precision:.1%}, recall: {recall:.1%}")
        avg_return = sum(dir_returns) / len(dir_returns) if dir_returns else 0.0
        print(f"Avg return per prediction: {avg_return:+.3f}%")

    return FoldResult(
        total,
        correct,
        dir_total,
        dir_correct,
        invalid_summary,
        dir_confusion,
        sum(dir_returns),
        len(dir_returns),
    )


def _parse_confidence(conf: str) -> float:
    """Parse predictor confidence into a 0-1 score."""
    if not conf:
        return 0.0
    conf = conf.strip().upper()
    if conf in {"LOW", "L"}:
        return 0.1
    if conf in {"MED", "MEDIUM", "M"}:
        return 0.5
    if conf in {"HIGH", "H"}:
        return 0.9
    if "/" in conf:
        parts = conf.split("/", 1)
        try:
            return float(parts[0]) / float(parts[1])
        except ValueError:
            return 0.0
    conf = conf.replace("%", "")
    try:
        val = float(conf)
    except ValueError:
        return 0.0
    return val / 100 if val > 1 else val


def _combine_predictions(
    primary: str,
    secondary: str,
    primary_conf: float,
    secondary_conf: float,
    policy: str,
) -> str:
    """Combine primary/secondary predictions with configurable tie-breaks."""
    if policy == "primary_wins":
        return primary
    if policy == "prefer_down":
        if primary == "DOWN" or secondary == "DOWN":
            return "DOWN"
        if primary == "UP" or secondary == "UP":
            return "UP"
        return "same"
    if policy == "confidence":
        if primary == secondary:
            return primary
        if primary_conf > secondary_conf:
            return primary
        if secondary_conf > primary_conf:
            return secondary
        return primary

    # primary_fallback (default)
    if primary == "same":
        return secondary
    if secondary == "same":
        return primary
    if primary == secondary:
        return primary
    return "same"


def _predict_for_config(
    predictor: GeminiPredictor,
    config: ExperimentConfig,
    hist: pd.DataFrame,
    hist_prices: np.ndarray,
    hist_volumes: np.ndarray | None,
    future_window: pd.DataFrame,
    last_price: float,
    training_examples: list[dict],
    htf_ts: np.ndarray | None,
    htf_rsi_vals: np.ndarray | None,
    sax_gate: set[str] | None,
    sax_potential_buckets: dict[str, tuple[int, int]] | None,
) -> tuple[str, float, int]:
    encoded, scale = do_encode(
        hist_prices,
        hist_volumes,
        config.encoding_type,
        config.add_patch_summary,
        config.patch_size,
        config.sax_paa_segments,
        config.sax_alphabet_size,
        config.sax_include_momentum,
        config.sax_include_range,
        config.sax_include_extremes,
        config.sax_include_rsi,
        config.sax_rsi_period,
        config.sax_use_arrows,
        config.sax_use_triangles,
        config.sax_use_range_symbols,
        config.sax_use_rsi_symbols,
        config.add_index_channel,
    )
    if htf_ts is not None and htf_rsi_vals is not None:
        htf_bucket = get_htf_rsi_bucket(
            htf_ts,
            htf_rsi_vals,
            hist["timestamp"].iloc[-1],
            config.sax_use_rsi_symbols,
        )
        encoded = f"{encoded}|HTF_RSI:{htf_bucket}"

    if config.encoding_type == "sax" and sax_gate is not None:
        base = sax_base_from_encoded(encoded)
        if base not in sax_gate:
            return "same", 0

    if config.encoding_type == "sax" and sax_potential_buckets is not None:
        base = sax_base_from_encoded(encoded)
        buckets = sax_potential_buckets.get(base)
        if buckets is None:
            token = "U?" if config.sax_potential_up_only else "U?D?"
        elif config.sax_potential_up_only:
            token = f"U{buckets[0]}"
        else:
            token = f"U{buckets[0]}D{buckets[1]}"
        encoded = f"{encoded}|POT:{token}"
    if config.encoding_type == "sax" and config.sax_future_shape:
        encoded = f"{encoded}|FUT:?"
    if config.encoding_type == "sax" and config.sax_prospect_buckets:
        encoded = f"{encoded}|PROS:U?D?"
    if config.encoding_type == "sax" and config.sax_prospect_time:
        encoded = f"{encoded}|PT:TU?TD?"
    if config.encoding_type == "sax" and config.sax_potential_top_n is not None:
        encoded = f"{encoded}|TOP:?"

    context_override = None
    if config.use_retrieval:
        if not config.use_retrieval_examples:
            neighbors = []
        elif config.use_sax_retrieval and config.encoding_type.startswith("sax"):
            neighbors = select_retrieval_examples_sax(
                training_examples,
                encoded,
                config.retrieval_k,
            )
        else:
            query_features = build_feature_vector(hist_prices, hist_volumes)
            neighbors = select_retrieval_examples(
                training_examples,
                query_features,
                config.retrieval_k,
            )
        context_override = format_training_context(neighbors, config)

    pred, conf, summary = predictor.predict(
        encoded,
        scale,
        last_price,
        context_override=context_override,
    )
    invalid_summary = 0
    if config.require_summary:
        expected = compute_summary(hist_prices, scale, config.encoding_type)
        if summary != expected:
            invalid_summary = 1
            pred = "same"
    return pred, _parse_confidence(conf), invalid_summary


def run_fold_ensemble(
    primary_predictor: GeminiPredictor,
    secondary_predictor: GeminiPredictor,
    test_df: pd.DataFrame,
    primary_config: ExperimentConfig,
    secondary_config: ExperimentConfig,
    fold_num: int,
    primary_examples: list[dict],
    secondary_examples: list[dict],
    primary_htf_ts: np.ndarray | None = None,
    primary_htf_rsi_vals: np.ndarray | None = None,
    secondary_htf_ts: np.ndarray | None = None,
    secondary_htf_rsi_vals: np.ndarray | None = None,
    primary_sax_gate: set[str] | None = None,
    secondary_sax_gate: set[str] | None = None,
    primary_sax_potential_buckets: dict[str, tuple[int, int]] | None = None,
    secondary_sax_potential_buckets: dict[str, tuple[int, int]] | None = None,
    ensemble_policy: str = "primary_fallback",
    show_last_trades: int = 0,
) -> FoldResult:
    print(f"\n--- Fold {fold_num + 1}/{primary_config.num_folds} ---")

    correct = 0
    total = 0
    dir_correct = 0
    dir_total = 0
    invalid_summary = 0
    dir_labels = ["UP", "DOWN", "same"]
    dir_confusion = {a: {p: 0 for p in dir_labels} for a in dir_labels}
    dir_returns: list[float] = []
    trades: list[dict] = []

    step = max(1, len(test_df) // primary_config.num_tests)

    for i in range(primary_config.window_size, len(test_df) - 1, step):
        if total >= primary_config.num_tests:
            break
        if i + 1 >= len(test_df):
            continue

        hist = test_df.iloc[i - primary_config.window_size : i]
        if primary_config.regime_min_atr_pct is not None:
            atr_pct = compute_atr_pct(hist.tail(primary_config.atr_lookback + 1))
            if atr_pct < primary_config.regime_min_atr_pct:
                continue
        future = test_df.iloc[i]
        horizon_end = min(len(test_df), i + primary_config.horizon)
        future_window = test_df.iloc[i:horizon_end]

        hist_prices = hist["close"].values
        hist_volumes = hist["volume"].values if "volume" in hist.columns else None
        future_price = future["close"]
        last_price = hist_prices[-1]
        change_pct = ((future_price - last_price) / last_price) * 100
        exit_price = future_window["close"].values[-1] if len(future_window) else last_price
        entry_time = hist["timestamp"].iloc[-1] if "timestamp" in hist.columns else None
        exit_time = (
            future_window["timestamp"].iloc[-1] if "timestamp" in future_window.columns else None
        )

        if primary_config.label_type == "triple_barrier":
            actual, change_pct = get_triple_barrier_label(
                hist_prices,
                future_window["close"].values,
                primary_config.barrier_up,
                primary_config.barrier_down,
            )
        elif primary_config.label_type == "hold_bucket":
            hold_label = get_hold_bucket_label(
                hist_prices,
                future_window["close"].values,
                primary_config.hold_buckets,
            )
            if hold_label is None:
                continue
            actual, change_pct = hold_label
        elif primary_config.label_type == "fixed_hold":
            fixed_label = get_fixed_hold_label(
                hist_prices,
                future_window["close"].values,
                primary_config.horizon,
            )
            if fixed_label is None:
                continue
            actual, change_pct = fixed_label
        elif primary_config.label_type == "trailing_barrier":
            actual, change_pct = get_trailing_barrier_label(
                hist_prices,
                future_window["close"].values,
                primary_config.barrier_up,
                primary_config.barrier_down,
            )
        else:
            actual = get_label(change_pct, primary_config.label_type)

        primary_pred, primary_conf, primary_invalid = _predict_for_config(
            primary_predictor,
            primary_config,
            hist,
            hist_prices,
            hist_volumes,
            future_window,
            last_price,
            primary_examples,
            primary_htf_ts,
            primary_htf_rsi_vals,
            primary_sax_gate,
            primary_sax_potential_buckets,
        )
        secondary_pred, secondary_conf, secondary_invalid = _predict_for_config(
            secondary_predictor,
            secondary_config,
            hist,
            hist_prices,
            hist_volumes,
            future_window,
            last_price,
            secondary_examples,
            secondary_htf_ts,
            secondary_htf_rsi_vals,
            secondary_sax_gate,
            secondary_sax_potential_buckets,
        )
        pred = _combine_predictions(
            primary_pred,
            secondary_pred,
            primary_conf,
            secondary_conf,
            ensemble_policy,
        )
        invalid_summary += primary_invalid + secondary_invalid

        is_right = is_correct(pred, actual, primary_config.label_type)
        if is_right:
            correct += 1
        total += 1

        if primary_config.label_type == "hold_bucket":
            dir_total += 1
            outcome = get_hold_bucket_outcome(
                pred,
                hist_prices,
                future_window["close"].values,
                primary_config.barrier_up,
                primary_config.barrier_down,
            )
            if outcome == "UP":
                dir_correct += 1
        elif is_directional(actual, primary_config.label_type):
            dir_total += 1
            if is_right:
                dir_correct += 1

        actual_dir = to_direction_label(
            actual,
            primary_config.label_type,
            hist_prices,
            future_window["close"].values,
            primary_config,
        )
        pred_dir = to_direction_label(
            pred,
            primary_config.label_type,
            hist_prices,
            future_window["close"].values,
            primary_config,
        )
        if actual_dir is not None and pred_dir is not None:
            dir_confusion[actual_dir][pred_dir] += 1
            if pred_dir == "UP":
                dir_returns.append(change_pct)
            elif pred_dir == "DOWN":
                dir_returns.append(-change_pct)
            else:
                dir_returns.append(0.0)

        status = "+" if is_right else "x"
        print(
            f"[{total}] {status} P:{pred:5} A:{actual:5} "
            f"Δ:{change_pct:+.2f}% (P1:{primary_pred}, P2:{secondary_pred})"
        )
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": last_price,
                "exit_price": exit_price,
                "pred": pred,
                "actual": actual,
                "change_pct": change_pct,
            }
        )

        time.sleep(0.5)

    if any(sum(v.values()) for v in dir_confusion.values()):
        for label in dir_labels:
            pred_total = sum(dir_confusion[a][label] for a in dir_labels)
            actual_total = sum(dir_confusion[label].values())
            tp = dir_confusion[label][label]
            precision = tp / pred_total if pred_total else 0.0
            recall = tp / actual_total if actual_total else 0.0
            print(f"{label} precision: {precision:.1%}, recall: {recall:.1%}")
        avg_return = sum(dir_returns) / len(dir_returns) if dir_returns else 0.0
        print(f"Avg return per prediction: {avg_return:+.3f}%")
    if show_last_trades:
        tail = trades[-show_last_trades:]
        print(f"\nLast {len(tail)} events:")
        for trade in tail:
            print(
                "  "
                f"{trade['entry_time']} -> {trade['exit_time']} | "
                f"{trade['entry_price']:.2f} -> {trade['exit_price']:.2f} | "
                f"{trade['pred']} / {trade['actual']} | "
                f"Δ:{trade['change_pct']:+.2f}%"
            )
        print("\nSimulated ladder (long-only, last events):")
        position = 0
        buys: list[float] = []
        realized_pnl = 0.0
        realized_cost = 0.0
        for trade in tail:
            if trade["pred"] == "UP":
                position += 1
                buys.append(float(trade["entry_price"]))
                action = f"BUY x1 (pos={position})"
            elif trade["pred"] == "DOWN":
                if position > 0:
                    sell_price = float(trade["exit_price"])
                    batch_cost = sum(buys)
                    batch_pnl = sum(sell_price - price for price in buys)
                    realized_pnl += batch_pnl
                    realized_cost += batch_cost
                    position = 0
                    buys = []
                    pct = (batch_pnl / batch_cost) if batch_cost else 0.0
                    action = f"SELL ALL @ {sell_price:.2f} (pnl={batch_pnl:+.2f}, {pct:+.2%})"
                else:
                    action = "SELL IGNORE"
            else:
                action = "HOLD"
            print(
                "  "
                f"{trade['entry_time']} | "
                f"{trade['pred']} -> {action}"
            )
        if realized_cost:
            total_pct = realized_pnl / realized_cost
        else:
            total_pct = 0.0
        print(
            f"\nLadder realized P&L: {realized_pnl:+.2f} "
            f"on cost {realized_cost:.2f} ({total_pct:+.2%})"
        )

    return FoldResult(
        total,
        correct,
        dir_total,
        dir_correct,
        invalid_summary,
        dir_confusion,
        sum(dir_returns),
        len(dir_returns),
    )


def _build_training_bundle(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
    htf_ts: np.ndarray | None,
    htf_rsi_vals: np.ndarray | None,
) -> tuple[list[dict], str, str, set[str] | None, dict[str, tuple[int, int]] | None]:
    sax_counts = None
    sax_top_set = None
    sax_potential_buckets = None
    if config.encoding_type == "sax":
        sax_counts = {}
        for _, row in train_df.iterrows():
            if len(sax_counts) > 5000:
                break
            idx = row.name
            if idx < config.window_size:
                continue
            window = train_df["close"].iloc[idx - config.window_size : idx].values
            base = _to_sax(window, config.sax_paa_segments, config.sax_alphabet_size)
            sax_counts[base] = sax_counts.get(base, 0) + 1
        if config.sax_potential_top_n is not None:
            sax_top_set = compute_sax_top_set(
                train_df["close"].values,
                config.window_size,
                config.sax_potential_horizon,
                config.sax_paa_segments,
                config.sax_alphabet_size,
                config.sax_potential_top_n,
            )
        if config.sax_potential_bucket_count is not None:
            sax_potential_buckets = compute_sax_potential_buckets(
                train_df["close"].values,
                config.window_size,
                config.sax_potential_horizon,
                config.sax_paa_segments,
                config.sax_alphabet_size,
                config.sax_potential_bucket_count,
            )
    examples = generate_examples(
        train_df,
        config,
        htf_ts,
        htf_rsi_vals,
        sax_counts,
        sax_potential_buckets,
    )
    if config.sax_potential_top_n is not None and sax_top_set is not None:
        for ex in examples:
            base = sax_base_from_encoded(ex["encoded"])
            flag = "1" if base in sax_top_set else "0"
            ex["encoded"] = f"{ex['encoded']}|TOP:{flag}"
    context = format_training_context(examples, config)
    base_context = format_training_context([], config)

    sax_gate = None
    if config.encoding_type == "sax" and config.sax_gate_top_n is not None:
        counts = {}
        for ex in examples:
            base = sax_base_from_encoded(ex["encoded"])
            counts[base] = counts.get(base, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[: config.sax_gate_top_n]
        sax_gate = {item[0] for item in top}

    return examples, context, base_context, sax_gate, sax_potential_buckets


def run_experiment(config: ExperimentConfig) -> dict:
    """Run full experiment with k-fold validation."""
    print("=" * 60)
    print(f"EXPERIMENT: {config}")
    print("=" * 60)

    # Load data
    df = load_data(config.symbol)
    htf_ts = None
    htf_rsi_vals = None
    if config.htf_symbol is not None:
        htf_df = load_data(config.htf_symbol)
        htf_ts, htf_rsi_vals = build_htf_rsi_lookup(
            htf_df,
            config.htf_rsi_period,
        )
    print(f"Loaded {len(df)} bars")

    # K-fold splits
    fold_size = len(df) // (config.num_folds + 1)
    results = []

    for fold in range(config.num_folds):
        # Split: use different test region for each fold
        test_start = fold_size * (fold + 1)
        test_end = test_start + fold_size

        train_df = pd.concat([df.iloc[:test_start], df.iloc[test_end:]])
        test_df = df.iloc[test_start:test_end]

        print(f"\nFold {fold + 1}: Train={len(train_df)}, Test={len(test_df)}")

        # Generate training
        (
            examples,
            context,
            base_context,
            sax_gate,
            sax_potential_buckets,
        ) = _build_training_bundle(
            train_df,
            config,
            htf_ts,
            htf_rsi_vals,
        )
        print(f"Training examples: {len(examples)}, Context: {len(context)} chars")

        if not config.use_chat and not config.use_retrieval and not config.use_cached_content:
            # Cache the static prompt prefix to reduce token costs.
            config.use_cached_content = True

        # Run predictions
        predictor = GeminiPredictor(context if not config.use_retrieval else base_context, config)
        try:
            predictor.start()
            result = run_fold(
                predictor,
                test_df,
                config,
                fold,
                examples,
                htf_ts,
                htf_rsi_vals,
                sax_gate,
                sax_potential_buckets,
            )
            results.append(result)
            print(
                f"Fold {fold + 1}: {result.accuracy:.1%} overall, "
                f"{result.directional_accuracy:.1%} directional, "
                f"summary_miss={result.invalid_summary}"
            )
        finally:
            predictor.cleanup()

        time.sleep(1)  # Between folds

    # Aggregate results
    total_correct = sum(r.correct for r in results)
    total_tests = sum(r.total for r in results)
    total_dir_correct = sum(r.directional_correct for r in results)
    total_dir_tests = sum(r.directional_total for r in results)
    dir_confusion = {"UP": {"UP": 0, "DOWN": 0, "same": 0},
                     "DOWN": {"UP": 0, "DOWN": 0, "same": 0},
                     "same": {"UP": 0, "DOWN": 0, "same": 0}}
    dir_return_sum = 0.0
    dir_return_count = 0
    for r in results:
        if r.dir_confusion:
            for a in dir_confusion:
                for p in dir_confusion[a]:
                    dir_confusion[a][p] += r.dir_confusion.get(a, {}).get(p, 0)
        dir_return_sum += r.dir_return_sum
        dir_return_count += r.dir_return_count

    summary = {
        "config": str(config),
        "overall_accuracy": total_correct / total_tests if total_tests > 0 else 0,
        "directional_accuracy": total_dir_correct / total_dir_tests if total_dir_tests > 0 else 0,
        "total_tests": total_tests,
        "directional_tests": total_dir_tests,
        "folds": config.num_folds,
        "summary_invalid": sum(r.invalid_summary for r in results),
    }
    if dir_return_count:
        summary["avg_return"] = dir_return_sum / dir_return_count
    if any(sum(v.values()) for v in dir_confusion.values()):
        summary["confusion"] = dir_confusion
        metrics = {}
        for label in ["UP", "DOWN", "same"]:
            pred_total = sum(dir_confusion[a][label] for a in dir_confusion)
            actual_total = sum(dir_confusion[label].values())
            tp = dir_confusion[label][label]
            precision = tp / pred_total if pred_total else 0.0
            recall = tp / actual_total if actual_total else 0.0
            metrics[label] = {"precision": precision, "recall": recall}
        summary["metrics"] = metrics

    print("\n" + "=" * 60)
    print(f"FINAL: {summary['overall_accuracy']:.1%} overall ({total_correct}/{total_tests})")
    print(
        f"       {summary['directional_accuracy']:.1%} directional "
        f"({total_dir_correct}/{total_dir_tests})"
    )
    print("=" * 60)

    return summary


def run_ensemble_experiment(
    primary_config: ExperimentConfig,
    secondary_config: ExperimentConfig,
    name: str,
    ensemble_policy: str = "primary_fallback",
    show_last_trades: int = 0,
) -> dict:
    """Run ensemble experiment by combining two configs."""
    if primary_config.label_type != secondary_config.label_type:
        raise ValueError("Ensemble configs must share label_type.")
    if primary_config.num_folds != secondary_config.num_folds:
        raise ValueError("Ensemble configs must share num_folds.")
    if primary_config.num_tests != secondary_config.num_tests:
        raise ValueError("Ensemble configs must share num_tests.")
    if primary_config.window_size != secondary_config.window_size:
        raise ValueError("Ensemble configs must share window_size.")
    if primary_config.horizon != secondary_config.horizon:
        raise ValueError("Ensemble configs must share horizon.")
    if primary_config.barrier_up != secondary_config.barrier_up:
        raise ValueError("Ensemble configs must share barrier_up.")
    if primary_config.barrier_down != secondary_config.barrier_down:
        raise ValueError("Ensemble configs must share barrier_down.")
    if primary_config.hold_buckets != secondary_config.hold_buckets:
        raise ValueError("Ensemble configs must share hold_buckets.")

    print("=" * 60)
    print(f"ENSEMBLE EXPERIMENT: {name}")
    print("=" * 60)
    print(f"Primary:   {primary_config}")
    print(f"Secondary: {secondary_config}")
    print(f"Policy:    {ensemble_policy}")

    df = load_data(primary_config.symbol)
    htf_ts_primary = None
    htf_rsi_primary = None
    if primary_config.htf_symbol is not None:
        htf_df = load_data(primary_config.htf_symbol)
        htf_ts_primary, htf_rsi_primary = build_htf_rsi_lookup(
            htf_df,
            primary_config.htf_rsi_period,
        )
    htf_ts_secondary = None
    htf_rsi_secondary = None
    if secondary_config.htf_symbol is not None:
        htf_df = load_data(secondary_config.htf_symbol)
        htf_ts_secondary, htf_rsi_secondary = build_htf_rsi_lookup(
            htf_df,
            secondary_config.htf_rsi_period,
        )
    print(f"Loaded {len(df)} bars")

    fold_size = len(df) // (primary_config.num_folds + 1)
    results = []

    for fold in range(primary_config.num_folds):
        test_start = fold_size * (fold + 1)
        test_end = test_start + fold_size

        train_df = pd.concat([df.iloc[:test_start], df.iloc[test_end:]])
        test_df = df.iloc[test_start:test_end]

        print(f"\nFold {fold + 1}: Train={len(train_df)}, Test={len(test_df)}")

        (
            primary_examples,
            primary_context,
            primary_base_context,
            primary_sax_gate,
            primary_sax_potential_buckets,
        ) = _build_training_bundle(
            train_df,
            primary_config,
            htf_ts_primary,
            htf_rsi_primary,
        )
        (
            secondary_examples,
            secondary_context,
            secondary_base_context,
            secondary_sax_gate,
            secondary_sax_potential_buckets,
        ) = _build_training_bundle(
            train_df,
            secondary_config,
            htf_ts_secondary,
            htf_rsi_secondary,
        )
        print(
            f"Primary examples: {len(primary_examples)} "
            f"(context {len(primary_context)} chars)"
        )
        print(
            f"Secondary examples: {len(secondary_examples)} "
            f"(context {len(secondary_context)} chars)"
        )

        if (
            not primary_config.use_chat
            and not primary_config.use_retrieval
            and not primary_config.use_cached_content
        ):
            primary_config.use_cached_content = True
        if (
            not secondary_config.use_chat
            and not secondary_config.use_retrieval
            and not secondary_config.use_cached_content
        ):
            secondary_config.use_cached_content = True

        primary_predictor = GeminiPredictor(
            primary_context if not primary_config.use_retrieval else primary_base_context,
            primary_config,
        )
        secondary_predictor = GeminiPredictor(
            secondary_context if not secondary_config.use_retrieval else secondary_base_context,
            secondary_config,
        )
        try:
            primary_predictor.start()
            secondary_predictor.start()
            result = run_fold_ensemble(
                primary_predictor,
                secondary_predictor,
                test_df,
                primary_config,
                secondary_config,
                fold,
                primary_examples,
                secondary_examples,
                htf_ts_primary,
                htf_rsi_primary,
                htf_ts_secondary,
                htf_rsi_secondary,
                primary_sax_gate,
                secondary_sax_gate,
                primary_sax_potential_buckets,
                secondary_sax_potential_buckets,
                ensemble_policy,
                show_last_trades,
            )
            results.append(result)
            print(
                f"Fold {fold + 1}: {result.accuracy:.1%} overall, "
                f"{result.directional_accuracy:.1%} directional, "
                f"summary_miss={result.invalid_summary}"
            )
        finally:
            primary_predictor.cleanup()
            secondary_predictor.cleanup()

        time.sleep(1)

    total_correct = sum(r.correct for r in results)
    total_tests = sum(r.total for r in results)
    dir_correct = sum(r.directional_correct for r in results)
    dir_total = sum(r.directional_total for r in results)
    summary_invalid = sum(r.invalid_summary for r in results)

    confusion = {label: {"UP": 0, "DOWN": 0, "same": 0} for label in ["UP", "DOWN", "same"]}
    for r in results:
        for actual in confusion:
            for pred in confusion[actual]:
                confusion[actual][pred] += r.dir_confusion[actual][pred]

    metrics = {}
    for label in confusion:
        pred_total = sum(confusion[a][label] for a in confusion)
        actual_total = sum(confusion[label].values())
        tp = confusion[label][label]
        metrics[label] = {
            "precision": (tp / pred_total) if pred_total else 0.0,
            "recall": (tp / actual_total) if actual_total else 0.0,
        }

    avg_return = 0.0
    total_returns = sum(r.dir_return_sum for r in results)
    total_return_count = sum(r.dir_return_count for r in results)
    if total_return_count:
        avg_return = total_returns / total_return_count

    return {
        "config": name,
        "overall_accuracy": total_correct / total_tests if total_tests else 0.0,
        "directional_accuracy": dir_correct / dir_total if dir_total else 0.0,
        "total_tests": total_tests,
        "directional_tests": dir_total,
        "folds": primary_config.num_folds,
        "summary_invalid": summary_invalid,
        "avg_return": avg_return,
        "confusion": confusion,
        "metrics": metrics,
    }


def run_best_ensemble() -> dict:
    """Run ensemble of the two top configs."""
    primary = ExperimentConfig(name="sax_symbols_all")
    secondary = replace(primary, name="sax_symbols_all_stride5", training_stride=5)
    return run_ensemble_experiment(
        primary,
        secondary,
        "ensemble_sax_symbols_all+stride5|policy=prefer_down",
        ensemble_policy="prefer_down",
    )


# =============================================================================
# MAIN
# =============================================================================


RESULTS_FILE = Path(__file__).parent / "results.json"


def save_result(result: dict) -> None:
    """Append result to JSON file."""
    import json

    results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    results.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved to {RESULTS_FILE}]")


def load_results() -> list[dict]:
    """Load all saved results."""
    import json

    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def run_experiments():
    """Run multiple experiments to find best config."""

    # Reduced scope: 1 fold, 40 tests each
    experiments = [
        ExperimentConfig(name="baseline", num_tests=40, num_folds=1),
        ExperimentConfig(
            name="signed_coord",
            num_tests=40,
            num_folds=1,
            encoding_type="signed_coordinate",
        ),
        ExperimentConfig(
            name="signed_coord_patch",
            num_tests=40,
            num_folds=1,
            encoding_type="signed_coordinate",
            add_patch_summary=True,
        ),
        ExperimentConfig(name="rank", num_tests=40, num_folds=1, encoding_type="rank"),
        ExperimentConfig(name="delta_bins", num_tests=40, num_folds=1, encoding_type="delta_bins"),
        ExperimentConfig(
            name="multichannel",
            num_tests=40,
            num_folds=1,
            encoding_type="multichannel",
        ),
        ExperimentConfig(
            name="multi_patch",
            num_tests=40,
            num_folds=1,
            encoding_type="multichannel",
            add_patch_summary=True,
        ),
        ExperimentConfig(
            name="multi_retrieval",
            num_tests=40,
            num_folds=1,
            encoding_type="multichannel",
            use_retrieval=True,
        ),
        ExperimentConfig(
            name="multi_retrieval_summary",
            num_tests=40,
            num_folds=1,
            encoding_type="multichannel",
            use_retrieval=True,
            require_summary=True,
        ),
        ExperimentConfig(
            name="signed_retrieval",
            num_tests=40,
            num_folds=1,
            encoding_type="signed_coordinate",
            use_retrieval=True,
        ),
        ExperimentConfig(
            name="rank_retrieval",
            num_tests=40,
            num_folds=1,
            encoding_type="rank",
            use_retrieval=True,
        ),
        ExperimentConfig(
            name="delta_retrieval",
            num_tests=40,
            num_folds=1,
            encoding_type="delta_bins",
            use_retrieval=True,
        ),
        ExperimentConfig(
            name="triple_barrier_retrieval",
            num_tests=40,
            num_folds=1,
            label_type="triple_barrier",
            horizon=6,
            barrier_up=0.25,
            barrier_down=0.25,
            encoding_type="multichannel",
            use_retrieval=True,
        ),
    ]

    all_results = load_results()
    completed = {r["config"] for r in all_results}

    for config in experiments:
        if str(config) in completed:
            print(f"[SKIP] {config.name} already done")
            continue
        try:
            result = run_experiment(config)
            result["timestamp"] = time.strftime("%Y-%m-%d %H:%M")
            all_results.append(result)
            save_result(result)
            time.sleep(2)
        except Exception as e:
            print(f"ERROR in {config.name}: {e}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'Config':<40} {'Overall':>10} {'Directional':>12}")
    print("-" * 64)
    for r in sorted(all_results, key=lambda x: x["directional_accuracy"], reverse=True):
        print(f"{r['config']:<40} {r['overall_accuracy']:>9.1%} {r['directional_accuracy']:>11.1%}")


if __name__ == "__main__":
    # Run full experiment suite
    run_experiments()

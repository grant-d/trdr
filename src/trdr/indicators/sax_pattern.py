"""SAX (Symbolic Aggregate approXimation) pattern recognition."""

import numpy as np

from ..data import Bar


def sax_pattern(bars: list[Bar], window: int = 20, segments: int = 5) -> str:
    """Convert price series to SAX symbolic pattern.

    Args:
        bars: List of OHLCV bars
        window: Lookback window for pattern
        segments: Number of PAA segments (alphabet size)

    Returns:
        SAX pattern string (e.g., "aabcd")
    """
    if len(bars) < window:
        return ""

    closes = np.array([b.close for b in bars[-window:]])

    # Z-normalize
    mean = np.mean(closes)
    std = np.std(closes)
    if std == 0:
        return "ccccc"  # Flat market
    normalized = (closes - mean) / std
    normalized = np.clip(normalized, -3, 3)  # Clip outliers

    # PAA: Piecewise Aggregate Approximation
    segment_size = window // segments
    paa = []
    for i in range(segments):
        start = i * segment_size
        end = start + segment_size
        paa.append(np.mean(normalized[start:end]))

    # Discretize to alphabet {a, b, c, d, e} using Gaussian breakpoints
    # Breakpoints for 5 symbols: -0.84, -0.25, 0.25, 0.84
    breakpoints = [-0.84, -0.25, 0.25, 0.84]
    alphabet = "abcde"
    pattern = ""
    for val in paa:
        idx = 0
        for bp in breakpoints:
            if val > bp:
                idx += 1
        pattern += alphabet[idx]

    return pattern


def sax_bullish_reversal(pattern: str) -> bool:
    """Detect bullish reversal patterns in SAX string.

    Balanced detection: identifies reversal patterns with quality filtering.

    Args:
        pattern: SAX pattern string

    Returns:
        True if bullish reversal pattern detected
    """
    if len(pattern) < 4:
        return False

    first_half = pattern[: len(pattern) // 2]
    second_half = pattern[len(pattern) // 2 :]

    # First half: bearish (mostly a/b)
    first_bearish = sum(1 for c in first_half if c in "ab") >= len(first_half) // 2

    # Second half: bullish (has d/e) and ends high
    second_bullish = sum(1 for c in second_half if c in "de") >= 1
    ends_high = pattern[-1] in "de"

    # Upward momentum in second half
    momentum = all(pattern[i] >= pattern[i - 1] for i in range(len(pattern) // 2, len(pattern)))

    return first_bearish and second_bullish and ends_high and momentum

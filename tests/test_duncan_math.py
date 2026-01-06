"""Tests for Duncan trailer math helpers."""

import math
from collections import deque

import numpy as np

from trdr.strategy.duncan.strategy import _ATR_WEIGHT, _atr_norm, _atr_weighted


def test_atr_weighted_matches_constants() -> None:
    value = _atr_weighted(10.0, 20.0)
    expected = (10.0 * _ATR_WEIGHT + 20.0 * (1.0 - _ATR_WEIGHT)) / 2.0
    assert math.isclose(value, expected, rel_tol=1e-9)


def test_atr_norm_median_plus_std() -> None:
    values = deque([1.0, 2.0, 3.0], maxlen=3)
    value = _atr_norm(values)
    expected = float(np.median(values) + np.std(values))
    assert math.isclose(value, expected, rel_tol=1e-9)

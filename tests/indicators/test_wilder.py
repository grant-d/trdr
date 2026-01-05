"""Tests for Wilder smoothing helpers."""

import math

from trdr.indicators.wilder import WilderEmaIndicator, wilder_ema_series


def test_wilder_indicator_seed_behavior() -> None:
    indicator = WilderEmaIndicator(3)
    assert math.isclose(indicator.update(1.0), 1.0)
    assert math.isclose(indicator.update(2.0), 2.0)
    # Seed complete: average of [1,2,3] = 2.0
    assert math.isclose(indicator.update(3.0), 2.0)
    # Wilder smoothing: (2*2 + 4)/3 = 2.666...
    assert math.isclose(indicator.update(4.0), 8.0 / 3.0)


def test_wilder_series_matches_indicator() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    series = wilder_ema_series(values, 3)
    assert len(series) == len(values)
    assert math.isclose(series[-2], 2.0)
    assert math.isclose(series[-1], 8.0 / 3.0)

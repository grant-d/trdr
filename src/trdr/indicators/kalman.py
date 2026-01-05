"""Kalman Filter for adaptive price smoothing."""

from ..data import Bar


def kalman(
    bars: list[Bar],
    measurement_noise: float = 0.1,
    process_noise: float = 0.01,
) -> float:
    """Calculate Kalman Filter for adaptive price smoothing.

    Uses 1D Kalman filter to smooth price data while adapting to trends.
    Lower measurement noise = more smoothing, higher = more responsive.

    Args:
        bars: List of OHLCV bars
        measurement_noise: Measurement uncertainty (R). Higher = noisier data
        process_noise: Process uncertainty (Q). Higher = more adaptive

    Returns:
        Current filtered price estimate
    """
    return KalmanIndicator.calculate(
        bars,
        measurement_noise=measurement_noise,
        process_noise=process_noise,
    )


def kalman_series(
    bars: list[Bar],
    measurement_noise: float = 0.1,
    process_noise: float = 0.01,
) -> list[float]:
    """Calculate Kalman Filter series for all bars.

    Args:
        bars: List of OHLCV bars
        measurement_noise: Measurement uncertainty (R)
        process_noise: Process uncertainty (Q)

    Returns:
        List of filtered price estimates
    """
    if not bars:
        return []

    filtered = []
    x_est = bars[0].close
    p_est = 1.0

    filtered.append(x_est)

    for bar in bars[1:]:
        # Prediction
        x_pred = x_est
        p_pred = p_est + process_noise

        # Update
        measurement = bar.close
        kalman_gain = p_pred / (p_pred + measurement_noise)
        x_est = x_pred + kalman_gain * (measurement - x_pred)
        p_est = (1 - kalman_gain) * p_pred

        filtered.append(x_est)

    return [float(x) for x in filtered]


class KalmanIndicator:
    """Streaming Kalman filter for price smoothing."""

    def __init__(self, measurement_noise: float = 0.1, process_noise: float = 0.01) -> None:
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self._x_est: float | None = None
        self._p_est = 1.0

    @staticmethod
    def calculate(
        bars: list[Bar],
        measurement_noise: float = 0.1,
        process_noise: float = 0.01,
    ) -> float:
        if not bars:
            return 0.0

        if len(bars) == 1:
            return bars[0].close

        x_est = bars[0].close
        p_est = 1.0

        for bar in bars[1:]:
            x_pred = x_est
            p_pred = p_est + process_noise
            measurement = bar.close
            kalman_gain = p_pred / (p_pred + measurement_noise)
            x_est = x_pred + kalman_gain * (measurement - x_pred)
            p_est = (1 - kalman_gain) * p_pred

        return float(x_est)

    def update(self, bar: Bar) -> float:
        if self._x_est is None:
            self._x_est = bar.close
            self._p_est = 1.0
            return float(self._x_est)

        x_pred = self._x_est
        p_pred = self._p_est + self.process_noise
        measurement = bar.close
        kalman_gain = p_pred / (p_pred + self.measurement_noise)
        self._x_est = x_pred + kalman_gain * (measurement - x_pred)
        self._p_est = (1 - kalman_gain) * p_pred
        return float(self._x_est)

    @property
    def value(self) -> float:
        return float(self._x_est) if self._x_est is not None else 0.0

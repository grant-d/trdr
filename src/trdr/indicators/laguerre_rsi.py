"""Laguerre RSI indicator for improved sensitivity."""

from ..data import Bar


def laguerre_rsi(bars: list[Bar], alpha: float = 0.2) -> float:
    """Calculate Laguerre RSI for improved sensitivity.

    Args:
        bars: List of OHLCV bars
        alpha: Smoothing factor (0-1, lower = more smoothing)

    Returns:
        Laguerre RSI value (0-100)
    """
    return LaguerreRsiIndicator.calculate(bars, alpha)


class LaguerreRsiIndicator:
    """Streaming Laguerre RSI calculator."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.gamma = 1.0 - alpha
        self._initialized = False
        self._L0 = 0.0
        self._L1 = 0.0
        self._L2 = 0.0
        self._L3 = 0.0
        self._value = 50.0

    def update(self, bar: Bar) -> float:
        close = bar.close
        if not self._initialized:
            self._L0 = close
            self._L1 = close
            self._L2 = close
            self._L3 = close
            self._initialized = True
            return self._value

        L0_new = (1 - self.gamma) * close + self.gamma * self._L0
        L1_new = -self.gamma * L0_new + self._L0 + self.gamma * self._L1
        L2_new = -self.gamma * L1_new + self._L1 + self.gamma * self._L2
        L3_new = -self.gamma * L2_new + self._L2 + self.gamma * self._L3

        self._L0, self._L1, self._L2, self._L3 = L0_new, L1_new, L2_new, L3_new

        cu = 0.0
        cd = 0.0
        if self._L0 > self._L1:
            cu += self._L0 - self._L1
        else:
            cd += self._L1 - self._L0
        if self._L1 > self._L2:
            cu += self._L1 - self._L2
        else:
            cd += self._L2 - self._L1
        if self._L2 > self._L3:
            cu += self._L2 - self._L3
        else:
            cd += self._L3 - self._L2

        if cu + cd == 0:
            self._value = 50.0
        else:
            self._value = float((cu / (cu + cd)) * 100)

        return self._value

    @staticmethod
    def calculate(bars: list[Bar], alpha: float = 0.2) -> float:
        if not bars:
            return 50.0
        calc = LaguerreRsiIndicator(alpha)
        for bar in bars:
            calc.update(bar)
        return calc.value

    @property
    def value(self) -> float:
        return float(self._value)

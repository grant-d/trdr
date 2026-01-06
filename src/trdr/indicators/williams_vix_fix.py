"""Williams Vix Fix volatility indicator for market bottoms."""

import numpy as np

from ..data import Bar


class WilliamsVixFixIndicator:
    """Streaming Williams Vix Fix calculator."""

    def __init__(
        self,
        pd: int = 22,
        bbl: int = 20,
        mult: float = 2.0,
        lb: int = 50,
        ph: float = 0.85,
    ) -> None:
        self.pd = max(1, pd)
        self.bbl = max(1, bbl)
        self.mult = mult
        self.lb = max(1, lb)
        self.ph = ph
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(
        bars: list[Bar],
        pd: int = 22,
        bbl: int = 20,
        mult: float = 2.0,
        lb: int = 50,
        ph: float = 0.85,
    ) -> tuple[float, str]:
        min_bars = max(pd, bbl, lb)
        if len(bars) < min_bars:
            return (0.0, "normal")

        recent_bars = bars[-pd:]
        closes = [b.close for b in recent_bars]
        highest_close = max(closes)
        current_low = bars[-1].low

        if highest_close == 0:
            return (0.0, "normal")

        wvf = ((highest_close - current_low) / highest_close) * 100.0

        wvf_history = []
        for i in range(len(bars) - lb, len(bars)):
            if i < pd:
                continue
            recent = bars[max(0, i - pd + 1) : i + 1]
            closes_i = [b.close for b in recent]
            highest_i = max(closes_i) if closes_i else 0
            low_i = bars[i].low
            if highest_i > 0:
                wvf_i = ((highest_i - low_i) / highest_i) * 100.0
                wvf_history.append(wvf_i)

        if len(wvf_history) < bbl:
            return (float(wvf), "normal")

        wvf_recent = wvf_history[-bbl:]
        mid_line = float(np.mean(wvf_recent))
        std_dev = float(np.std(wvf_recent))
        upper_band = mid_line + (mult * std_dev)

        range_high = max(wvf_history) * ph

        if wvf >= upper_band or wvf >= range_high:
            alert_state = "high"
        else:
            alert_state = "normal"

        return (float(wvf), alert_state)

    def update(self, bar: Bar) -> tuple[float, str]:
        self._bars.append(bar)
        return self.calculate(
            self._bars,
            pd=self.pd,
            bbl=self.bbl,
            mult=self.mult,
            lb=self.lb,
            ph=self.ph,
        )

    @property
    def value(self) -> tuple[float, str]:
        if not self._bars:
            return (0.0, "normal")
        return self.calculate(
            self._bars,
            pd=self.pd,
            bbl=self.bbl,
            mult=self.mult,
            lb=self.lb,
            ph=self.ph,
        )

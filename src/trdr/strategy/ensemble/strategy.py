"""Ensemble strategy using decision tree trained on multiple indicators.

Combines multiple technical indicators using a decision tree classifier:
- Lorentzian Classifier: Advanced ML-based price direction prediction
- SMI: Stochastic Momentum Index
- RSI: Relative Strength Index
- Supertrend: Trend direction indicator
- ADX: Trend strength indicator
- ATR: Market volatility regime

The decision tree is trained on historical data to learn optimal entry patterns.
Exits are managed via ATR-based trailing stops, profit targets, and trend reversals.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ...data import Bar
from ...indicators import (
    AdaptiveSupertrendIndicator,
    AdxIndicator,
    AtrIndicator,
    BollingerBandsIndicator,
    LorentzianClassifierIndicator,
    MssIndicator,
    OrderFlowImbalanceIndicator,
    RsiIndicator,
    SmiIndicator,
    SupertrendIndicator,
    VolatilityRegimeIndicator,
    WilliamsVixFixIndicator,
)
from ..base_strategy import BaseStrategy, StrategyConfig
from ..types import DataRequirement, Position, Signal, SignalAction


@dataclass(frozen=True)
class EnsembleConfig(StrategyConfig):
    """Configuration for ensemble random forest strategy.

    Args:
        symbol: Trading symbol (inherited from StrategyConfig)
        timeframe: Bar timeframe (inherited from StrategyConfig)
        lookback_bars: Bars for feature calculation
        retrain_every: Retrain model every N bars (0 = never retrain)
        n_estimators: Number of trees in random forest
        max_tree_depth: Maximum depth of each tree
        min_samples_split: Minimum samples to split tree node
    """

    lookback_bars: int = 2000  # More history for the initial model
    retrain_every: int = 240   # Periodic retraining
    n_estimators: int = 100
    max_tree_depth: int = 10
    min_samples_split: int = 50


class EnsembleStrategy(BaseStrategy):
    """Ensemble strategy using random forest on multiple indicators.

    Features:
    - Lorentzian Classifier (ML-based direction prediction)
    - SMI (Stochastic momentum)
    - RSI (Standard momentum)
    - Supertrend (Trend direction)
    - ADX (Trend strength)
    - ATR (Volatility scale)

    Exits:
    - Trailing stop: 2.5 * ATR
    - Take profit: 4.0 * ATR
    - Trend flip: Exit if Supertrend turns bearish
    """

    def __init__(self, config: EnsembleConfig):
        """Initialize with config."""
        super().__init__(config, name="Ensemble")
        self.config: EnsembleConfig = config
        self._reset_state()
        self._model: RandomForestClassifier | None = None
        self._model_path = Path(__file__).parent / "ensemble_model.pkl"
        self._bars_since_train = 0
        self._load_or_init_model()
        self._entry_price: float = 0.0
        self._trailing_stop: float = 0.0

    def _reset_state(self) -> None:
        """Reset streaming indicators."""
        self._lorentzian = LorentzianClassifierIndicator(neighbors=8, max_bars_back=200)
        self._smi = SmiIndicator(k=10, d=3)
        self._rsi = RsiIndicator(period=14)
        self._supertrend = SupertrendIndicator(period=10, multiplier=3.0)
        self._adx = AdxIndicator(period=14)
        self._volatility = VolatilityRegimeIndicator(lookback=50)
        self._mss = MssIndicator(lookback=20)
        self._bb = BollingerBandsIndicator(period=20, std_mult=2.0)
        self._wvf = WilliamsVixFixIndicator()
        self._ast = AdaptiveSupertrendIndicator()
        self._ofi = OrderFlowImbalanceIndicator(lookback=5)
        self._atr = AtrIndicator(period=14)
        self._last_index = 0
        self._feature_history: list[list[float]] = []
        self._label_history: list[int] = []

    def reset(self) -> None:
        """Reset strategy state for new run."""
        self._reset_state()
        self._bars_since_train = 0
        self._entry_price = 0.0
        self._trailing_stop = 0.0

    def _load_or_init_model(self) -> None:
        """Load trained model or initialize new one."""
        if self._model_path.exists():
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)
        else:
            self._model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_tree_depth,
                min_samples_split=self.config.min_samples_split,
                random_state=42,
                n_jobs=-1,
            )

    def _save_model(self) -> None:
        """Save trained model to disk."""
        if self._model:
            with open(self._model_path, "wb") as f:
                pickle.dump(self._model, f)

    def _extract_features(self, bars: list[Bar]) -> np.ndarray:
        """Extract feature vector from current indicators.

        Returns:
            Feature array: [lorentzian, smi, rsi, st_dir, adx, vol_numeric, mss, bb_dist, wvf, ast_dir, ofi, atr_norm]
        """
        if len(bars) < 1:
            return np.zeros(12)

        current_bar = bars[-1]

        # Update all indicators to current bar
        lorentzian_val = self._lorentzian.update(current_bar)
        smi_val = self._smi.update(current_bar)
        rsi_val = self._rsi.update(current_bar)
        st_result = self._supertrend.update(current_bar)
        adx_val = self._adx.update(current_bar)
        vol_regime = self._volatility.update(current_bar)
        mss_val = self._mss.update(current_bar)
        bb_result = self._bb.update(current_bar)
        wvf_result = self._wvf.update(current_bar)
        ast_result = self._ast.update(current_bar)
        ofi_val = self._ofi.update(current_bar)
        atr_val = self._atr.update(current_bar)

        # WVF calculation (stateful inside the indicator now if implemented correctly, but let's use the instance)
        # Note: WilliamsVixFixIndicator.update returns (value, alert_state)
        # We need to make sure _wvf is initialized. I'll add it to _reset_state in next step if missed.
        # Checking previous step... I missed _wvf in _reset_state. I'll fix that.
        
        # Supertrend direction: 1 = bullish, -1 = bearish
        st_dir = st_result[1] if isinstance(st_result, tuple) else 0.0

        # Volatility regime as numeric: "low" = -1, "medium" = 0, "high" = 1
        vol_map = {"low": -1.0, "medium": 0.0, "high": 1.0}
        vol_numeric = vol_map.get(vol_regime, 0.0)

        # Bollinger Band distance: (price - middle) / middle
        bb_dist = (current_bar.close - bb_result[1]) / bb_result[1] if bb_result[1] != 0 else 0.0

        # AST direction
        ast_dir = ast_result[1] if isinstance(ast_result, tuple) else 0.0

        # ATR normalized by price
        atr_norm = atr_val / current_bar.close if current_bar.close > 0 else 0.0

        return np.array([
            float(lorentzian_val),
            float(smi_val),
            float(rsi_val if rsi_val is not None else 50.0) / 100.0,
            float(st_dir),
            float(adx_val) / 100.0,
            float(vol_numeric),
            float(mss_val) / 100.0,
            float(bb_dist),
            float(wvf_result[0]) / 10.0,
            float(ast_dir),
            float(ofi_val),
            float(atr_norm),
        ])

    def _generate_labels(self, bars: list[Bar], lookahead: int = 12) -> list[int]:
        """Generate training labels: 1 = profitable long, 0 = no trade.

        Labeling: 0.5% profit in 12 bars (3 hours).
        """
        labels = []
        for i in range(len(bars) - lookahead):
            entry_price = bars[i].close
            future_prices = [bars[j].close for j in range(i + 1, min(i + lookahead + 1, len(bars)))]
            if not future_prices:
                labels.append(0)
                continue

            max_future = max(future_prices)
            profit_pct = (max_future - entry_price) / entry_price
            labels.append(1 if profit_pct > 0.005 else 0)

        labels.extend([0] * lookahead)
        return labels

    def _train_model(self, bars: list[Bar]) -> None:
        """Train decision tree on historical bars."""
        if len(bars) < self.config.lookback_bars:
            return

        print(f"DEBUG: Starting model training on {len(bars)} bars...")
        self._reset_state()
        features = []
        for bar in bars:
            feat = self._extract_features([bar])
            features.append(feat)

        labels = self._generate_labels(bars)

        if len(features) > self.config.min_samples_split and len(features) == len(labels):
            features_array = np.array(features[: len(labels)])
            labels_array = np.array(labels)

            valid_mask = ~np.isnan(features_array).any(axis=1)
            features_array = features_array[valid_mask]
            labels_array = labels_array[valid_mask]

            if len(features_array) > self.config.min_samples_split:
                self._model.fit(features_array, labels_array)
                self._save_model()
                print("DEBUG: Model training complete.")

        self._reset_state()
        for bar in bars:
            self._extract_features([bar])

    def get_data_requirements(self) -> list[DataRequirement]:
        """Declare data feeds for this strategy."""
        return [
            DataRequirement(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                lookback=self.config.lookback,
                role="primary",
            ),
        ]

    def generate_signal(
        self,
        bars: dict[str, list[Bar]],
        position: Position | None,
    ) -> Signal:
        """Generate trading signal using ensemble decision tree."""
        primary_key = f"{self.config.symbol}:{self.config.timeframe}"
        primary_bars = bars[primary_key]

        if len(primary_bars) < self.config.lookback_bars:
            return Signal(
                action=SignalAction.HOLD,
                price=primary_bars[-1].close if primary_bars else 0,
                confidence=0.0,
                reason="Insufficient data",
            )

        current_price = primary_bars[-1].close

        if self._model is None or not hasattr(self._model, "estimators_"):
            self._train_model(primary_bars)
            return Signal(action=SignalAction.HOLD, price=current_price, confidence=0.0, reason="Training")

        self._bars_since_train += 1
        if self.config.retrain_every > 0 and self._bars_since_train >= self.config.retrain_every:
            self._train_model(primary_bars)
            self._bars_since_train = 0

        if self._last_index > len(primary_bars):
            self._reset_state()
        for bar in primary_bars[self._last_index :]:
            self._extract_features([bar])
        self._last_index = len(primary_bars)

        features = self._extract_features([primary_bars[-1]])
        st_dir = features[3]
        atr_val = self._atr.value

        features_row = features.reshape(1, -1)
        prediction = self._model.predict(features_row)[0]
        probabilities = self._model.predict_proba(features_row)[0]
        confidence = float(probabilities[1]) # Always use long probability for signal

        if position and position.side == "long":
            # Emergency Confidence Exit or Trend Flip
            if confidence < 0.4 or st_dir == -1:
                 return Signal(action=SignalAction.CLOSE, price=current_price, confidence=1.0, reason="Confidence exit")

            self._trailing_stop = max(self._trailing_stop, current_price - 2.5 * atr_val)
            profit_target = self._entry_price + 4.5 * atr_val

            if current_price < self._trailing_stop:
                return Signal(action=SignalAction.CLOSE, price=current_price, confidence=1.0, reason="Stop loss")
            
            if current_price > profit_target:
                 return Signal(action=SignalAction.CLOSE, price=current_price, confidence=1.0, reason="Take profit")

            return Signal(action=SignalAction.HOLD, price=current_price, confidence=0.5, reason="Holding")

        if not position:
            # Entry condition: Model bullish AND Supertrend bullish
            if prediction == 1 and confidence > 0.55 and st_dir == 1:
                self._entry_price = current_price
                self._trailing_stop = current_price - 2.5 * atr_val
                return Signal(
                    action=SignalAction.BUY,
                    price=current_price,
                    confidence=confidence,
                    reason=f"Entry signal (conf={confidence:.2f})",
                )

        return Signal(action=SignalAction.HOLD, price=current_price, confidence=0.0, reason="No signal")

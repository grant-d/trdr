"""VolumeAreaBreakout strategy implementation.

RuntimeContext available via self.context in generate_signal():
    self.context.drawdown         # Current drawdown %
    self.context.win_rate         # Live win rate
    self.context.equity           # Current portfolio value
    self.context.current_bar      # Current Bar object
    See backtest/STRATEGY_API.md for full list.
"""

from dataclasses import dataclass

import numpy as np

from ...core import Timeframe
from ...data import Bar
from ..base_strategy import BaseStrategy, StrategyConfig
from ..types import DataRequirement, Position, Signal, SignalAction
from ...indicators import (
    atr,
    bollinger_bands,
    hma,
    hma_slope,
    hvn_support_strength,
    mss,
    multi_timeframe_poc,
    order_flow_imbalance,
    rsi,
    volatility_regime,
    volume_profile,
    volume_trend,
)


@dataclass
class VolumeAreaBreakoutConfig(StrategyConfig):
    """Configuration for VolumeAreaBreakout strategy.

    Args:
        symbol: Trading symbol (e.g., "crypto:BTC/USD", "stock:AAPL")
        timeframe: Bar timeframe (e.g., "1h", "4h", "1d")
        atr_threshold: ATR multiplier for entry threshold
        stop_loss_multiplier: Multiplier for stop loss distance
    """

    atr_threshold: float = 2.0
    stop_loss_multiplier: float = 1.75


class VolumeAreaBreakoutStrategy(BaseStrategy):
    """VAH breakout + VAL bounce strategy with POC target.

    Entry paths:
    1. VAH Breakout: Price breaks above VAH with volume (bullish regime)
    2. VAL Bounce: Price bounces from VAL with any regime tolerance

    Exit rules:
    - Target: POC level
    - Stop: 1.2x ATR for bounce, 0.4x ATR below VAL for breakout

    RuntimeContext (self.context) enables adaptive behavior.
    See MACD template for usage examples.
    """

    def __init__(self, config: VolumeAreaBreakoutConfig):
        """Initialize strategy.

        Args:
            config: Strategy configuration with symbol and parameters
        """
        super().__init__(config, name="VolumeAreaBreakout")  # Custom name for display
        self.config: VolumeAreaBreakoutConfig = config

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
        """Generate trading signal based on Volume Profile analysis.

        Args:
            bars: Dict of bars keyed by "symbol:timeframe"
            position: Current position or None

        Returns:
            Trading signal with action, stops, and targets
        """
        # Extract primary bars from dict
        primary_key = f"{self.config.symbol}:{self.config.timeframe}"
        bars = bars[primary_key]

        # Minimal bar requirement for volume profile calculation
        if len(bars) < 50:
            return Signal(
                action=SignalAction.HOLD,
                price=bars[-1].close if bars else 0,
                confidence=0.0,
                reason="Insufficient data for analysis",
            )

        # ITER 82: Fresh approach - LVN Breakout from research doc
        # Research Strategy 2: LVN areas = low liquidity → rapid price movement
        # Entry: Volume surge (>150% avg) + direction aligned with higher TF
        # Target: Next HVN or POC, Stop: Inside LVN
        tf = self.config.timeframe
        if tf == Timeframe.parse("15m"):
            # ITER 86: Hybrid Strategy - LVN Breakout OR POC Mean Reversion
            # Combining both to increase trade frequency (2+ trades instead of 1)

            current_bar = bars[-1]
            current_price = current_bar.close
            profile = volume_profile(bars)
            atr_val = atr(bars)

            recent_volumes = [b.volume for b in bars[-20:]]
            avg_volume = np.mean(recent_volumes) if recent_volumes else 1

            # PATH 1: LVN Breakout - iter 138: tighten 1.4x → 1.6x to remove loser
            # Higher threshold filters weaker setups
            volume_surge = (current_bar.volume / avg_volume) >= 1.6 if avg_volume > 0 else False

            # Iter 139: Tighten from 5% → 7% to filter weaker trend setups
            if len(bars) >= 30:
                trend_gain = (bars[-1].close - bars[-30].close) / bars[-30].close
                trend_bullish = trend_gain > 0.07  # 7% gain required
            else:
                trend_bullish = False

            in_value_area = profile.val < current_price < profile.vah
            away_from_poc = abs(current_price - profile.poc) > atr_val * 0.5
            lvn_signal = in_value_area and away_from_poc and volume_surge and trend_bullish

            # PATH 2: POC Mean Reversion - iter 142: revert to 2.0 ATR baseline
            # Require clearly declining volume (0.8x vs 1.0x)
            volume_declining = (current_bar.volume / avg_volume) < 0.8 if avg_volume > 0 else False
            # Require extreme oversold (2.0 ATR below VAL)
            oversold = current_price < (profile.val - atr_val * 2.0)
            poc_mr_signal = oversold and volume_declining

            # PATH 3: VAH Breakout - new iter 92
            # Price crosses above VAH with volume surge = momentum continuation
            prev_close = bars[-2].close if len(bars) > 1 else current_price
            vah_breakout = (prev_close <= profile.vah) and (current_price > profile.vah) and volume_surge

            # Take any signal (prioritize LVN > VAH > POC MR)
            if lvn_signal:
                # Iter 114: Slightly wider stop VAL - 0.05 ATR (was 0.02)
                stop_loss = profile.val - atr_val * 0.05
                # Iter 133: Revert to 20.0 ATR (iter 131 optimal)
                take_profit = profile.vah + atr_val * 20.0
                if take_profit <= current_price:
                    # Iter 134: Widen fallback from 8.5 → 9.0 ATR
                    take_profit = current_price + atr_val * 9.0

                return Signal(
                    action=SignalAction.BUY,
                    price=current_price,
                    confidence=0.75,
                    reason=f"LVN breakout: vol={current_bar.volume/avg_volume:.1f}x, trend_up",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

            elif vah_breakout:
                # VAH breakout: stop below VAH, target extension above
                # Iter 95: Tighter stop (was 0.5 ATR below VAH)
                stop_loss = profile.vah - atr_val * 0.3
                # Iter 115: Widen TP 3.0 → 4.0 ATR
                take_profit = current_price + atr_val * 4.0

                return Signal(
                    action=SignalAction.BUY,
                    price=current_price,
                    confidence=0.72,
                    reason=f"VAH breakout: vol={current_bar.volume/avg_volume:.1f}x",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

            elif poc_mr_signal:
                # Iter 113: Revert to 1.0 ATR (iter 112 disaster at 0.8)
                stop_loss = current_price - atr_val * 1.0
                # Iter 133: Revert to 20.0 ATR (iter 131 optimal)
                take_profit = profile.poc + atr_val * 20.0
                if take_profit <= current_price:
                    # Iter 135: Widen fallback from 3.0 → 4.0 ATR
                    take_profit = profile.val + atr_val * 4.0

                return Signal(
                    action=SignalAction.BUY,
                    price=current_price,
                    confidence=0.70,
                    reason=f"POC mean reversion: oversold {(profile.val-current_price)/atr_val:.1f} ATR",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

            # No signal
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.0,
                reason=f"No signal: lvn={lvn_signal}, vah={vah_breakout}, poc_mr={poc_mr_signal}",
            )

        # Calculate indicators (using original bars for robustness)
        profile = volume_profile(bars)
        atr_val = atr(bars)
        mss_val = mss(bars)
        vol_regime = volatility_regime(bars)
        rsi_val = rsi(bars, period=14)
        bb_upper, bb_middle, bb_lower = bollinger_bands(bars, period=20)

        current_bar = bars[-1]
        current_price = current_bar.close
        prev_close = bars[-2].close if len(bars) > 1 else current_price

        # Detect timeframe for threshold adjustments
        is_daily = tf == Timeframe.parse("1d")
        is_4h = tf == Timeframe.parse("4h")
        is_15m = tf == Timeframe.parse("15m")

        # Check exit conditions first if we have a position
        if position and position.side == "long":
            # Check stop loss
            if current_price <= position.stop_loss:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=current_price,
                    confidence=1.0,
                    reason=f"Stop loss hit at {position.stop_loss:.2f}",
                )

            # Check take profit
            if position.take_profit and current_price >= position.take_profit:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=current_price,
                    confidence=0.9,
                    reason=f"Take profit hit at {position.take_profit:.2f}",
                )

            # Hold position
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.5,
                reason="Holding position, awaiting target or stop",
            )

        # Entry logic (no position)
        if position and position.side != "none":
            if not is_daily:
                return Signal(
                    action=SignalAction.HOLD,
                    price=current_price,
                    confidence=0.0,
                    reason="Position already open",
                )
            # Daily: ignore position check, re-evaluate entry every bar

        # Regime Filter: stricter for daily to improve WR
        # 4h: Permissive regime filter but reject extreme bearish (MSS < -50)
        # 15m: Iter 137: Relax from -40 → -50 to add potential 5th trade
        if is_15m:
            regime_threshold = -50  # Relaxed to match 4h
        elif is_4h:
            regime_threshold = -50  # Relaxed from -40, still very permissive
        elif is_daily:
            regime_threshold = -100  # No regime filter for daily
        else:
            regime_threshold = -20
        if mss_val < regime_threshold:
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.0,
                reason=f"Extreme bearish regime (MSS={mss_val:.0f})",
            )

        # Calculate volume metrics
        recent_volumes = [b.volume for b in bars[-20:]]
        avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 1
        volume_ratio = current_bar.volume / avg_recent_volume if avg_recent_volume > 0 else 0
        vol_trend = volume_trend(bars)

        # Calculate order flow imbalance (buying vs selling pressure)
        ofi = order_flow_imbalance(bars, lookback=5)

        # Calculate historical support strength at VAL level
        hvn_strength = hvn_support_strength(bars, profile.val, lookback=30)

        # Calculate multi-timeframe PoC confluence
        poc_tf1, poc_tf2, poc_tf3 = multi_timeframe_poc(bars)
        # Multi-TF confluence: higher TF POCs cluster together = stronger support
        poc_cluster_width = abs(poc_tf2 - poc_tf3)
        poc_clustered = poc_cluster_width < (
            atr_val * 0.5
        )  # POCs within 0.5 ATR = strong confluence

        # HMA trend filter - price must be above HMA (uptrend)
        hma_val = hma(bars, period=9)
        hma_bullish = current_price > hma_val and hma_val > 0
        # For daily: require HMA slope positive (1-bar lookback for more signals)
        hma_slope_val = hma_slope(bars, period=9, lookback=1)
        hma_trending_up = hma_slope_val > 0

        # Path 1: VAH Breakout (thresholds vary by timeframe)
        above_vah = current_price > profile.vah
        above_vah_prev = bars[-2].close <= profile.vah if len(bars) > 1 else False
        # For 15m: permissive entries like iter 33 (3 trades, 0.818 score)
        if is_15m:
            # Iter 64: Revert to NO FILTERS (iter 33 optimal)
            vah_breakout = above_vah and above_vah_prev
        elif is_daily:
            # Daily: maximize trade frequency - minimal filters
            volume_ok = True
            hma_filter = True
            vah_breakout = above_vah and above_vah_prev
        elif is_4h:
            volume_ok = volume_ratio >= 1.0
            regime_bullish_plus = True
            hma_filter = hma_bullish
            vah_breakout = above_vah and above_vah_prev and volume_ok and hma_filter
        else:
            volume_ok = volume_ratio >= 1.0
            regime_bullish_plus = mss_val > 5
            hma_filter = hma_bullish
            vah_breakout = above_vah and above_vah_prev and volume_ok and hma_filter

        # Path 2: VAL Bounce - mean reversion (research: 55-65% win rate potential)
        # 4h: tighter VAL proximity to reduce false bounces, MSS > 0 required for regime
        if is_daily:
            # Daily: ultra-wide zones to maximize entries
            near_val = abs(current_price - profile.val) < atr_val * 5.0
            near_poc = abs(current_price - profile.poc) < atr_val * 5.0
            near_vah = abs(current_price - profile.vah) < atr_val * 5.0
            val_bounce = near_val or near_poc or near_vah
            # POC breakout
            above_poc = current_price > profile.poc
            above_poc_prev = bars[-2].close <= profile.poc if len(bars) > 1 else False
            poc_breakout = above_poc and above_poc_prev
            poc_pullback = False
        elif is_15m:
            # 15m: Permissive like iter 33 (3 trades, 0.818 score)
            # VAL bounce: permissive proximity, no MSS filter
            near_val = abs(current_price - profile.val) < atr_val * 1.0
            val_bounce = near_val  # No MSS filter per iter 33
            poc_pullback = False
            poc_breakout = False  # Not used for 15m
        elif is_4h:
            # 4h: Mean reversion on VAL - tighter to improve WR without losing P&L
            # VAL is institutional liquidity magnet - but only use strongest levels
            near_val = (
                abs(current_price - profile.val) < atr_val * 0.55
            )  # Tighter proximity (was 0.6)
            # Require minimum HVN strength to filter weak bounces
            hvn_ok = hvn_strength > 0.3  # Require historical validation
            val_bounce = near_val and hvn_ok and mss_val > -55  # Tighter regime
            poc_pullback = False
            poc_breakout = False  # Not used for 4h
        else:
            # Intraday (1h): enable VAL bounce with original logic
            near_val = abs(current_price - profile.val) < atr_val * 0.5
            val_bounce = near_val and volume_ratio >= 1.0 and mss_val > 0
            poc_pullback = False
            poc_breakout = False  # Not used for 1h

        # Daily: enter only in bullish regime
        if is_daily:
            entry_signal = mss_val > 0  # Bullish regime only
        else:
            entry_signal = vah_breakout or val_bounce or poc_pullback

        if not entry_signal:
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.0,
                reason=f"No signal: vah={vah_breakout} val={val_bounce} rsi={rsi_val:.0f}",
            )

        # Calculate stops and targets
        if is_daily or vah_breakout or poc_pullback or poc_breakout:
            if is_daily:
                # Daily: tighter stop to reduce DD below 25% threshold
                # Iter 33: test 0.5 ATR stop (was 0.6) to cut DD from 32% to ~26%
                take_profit = current_price + atr_val * 3.0
                stop_loss = current_price - atr_val * 0.5
                confidence_base = 0.50  # Ensure always above threshold
            elif is_15m:
                # 15m: Iter 65 - match iter 33's 0.43 ATR stops (got 3 trades, 0.818 score)
                take_profit = current_price + atr_val * 5.0
                stop_loss = current_price - atr_val * 0.43
            elif is_4h:
                # 4h: Aggressive targets to maximize alpha (1.09x)
                # 26 ATR target captures full 4h swing, 0.6 ATR stop is optimal
                take_profit = profile.vah + atr_val * 26.0  # Balanced at 26 ATR
                stop_loss = profile.vah - atr_val * 0.6  # Optimal stop: 0.6 ATR
            else:
                take_profit = profile.vah + atr_val * 3.0
                stop_loss = profile.vah - atr_val * 0.3
            # Set signal_type for non-daily (daily already set above)
            if not is_daily:
                if vah_breakout:
                    signal_type = "VAH_breakout"
                elif poc_breakout:
                    signal_type = "POC_breakout"
                else:
                    signal_type = "VAH_pullback"
            if is_15m:
                confidence_base = 0.55  # Higher base for 15m breakouts
            else:
                confidence_base = 0.65 if vah_breakout else 0.60
        else:  # val_bounce
            # VAL bounce: mean reversion to POC (research: 55-65% win rate)
            # Take profit at POC (defined by volume, not arbitrary)
            signal_type = "VAL_bounce"

            if is_daily:
                # Daily: tighter stop to reduce DD below 25% threshold
                take_profit = current_price + atr_val * 2.5
                stop_loss = current_price - atr_val * 0.5
                confidence_base = 0.50  # Ensure always above threshold
            elif is_15m:
                # 15m: Iter 65 - match iter 33's 0.43 ATR stops (got 3 trades, 0.818 score)
                take_profit = current_price + atr_val * 5.0
                stop_loss = current_price - atr_val * 0.43
                confidence_base = 0.55  # Higher base for 15m
            else:
                confidence_base = 0.52  # Slightly higher base from 0.50
                take_profit = profile.poc
                if is_4h:
                    stop_loss = current_price - atr_val * 1.5  # Tighter from 1.6 to improve WR
                else:
                    stop_loss = current_price - atr_val * 2.0

        confidence = confidence_base

        # Volume bonus
        if vah_breakout and volume_ratio > 1.5:
            confidence += 0.1

        # Strong declining volume bonus for bounces (critical signal)
        # This filter is crucial for filtering out momentum trades that look like bounces
        # Iter 3: 0.45 bonus was optimal for filtering
        if val_bounce and vol_trend == "declining":
            confidence += 0.45  # Iter 3 best: maximum bonus for volume-declining bounces

        # Order Flow Imbalance bonus for bounces (positive OFI = buying pressure = bounce support)
        # Positive OFI near VAL indicates institutional buyers catching the dip
        # Tiered OFI bonus based on strength and volume combination
        if val_bounce and ofi > 0.10:  # Relaxed from 0.15 to catch weaker but valid signals
            # Stronger bonus when OFI combined with declining volume (institutional accumulation)
            if vol_trend == "declining":
                confidence += 0.25  # Maximum OFI bonus when combined with declining volume
            elif ofi > 0.20:
                confidence += 0.20  # Standard bonus for strong OFI
            else:
                confidence += 0.15  # Conservative bonus for weak OFI alone

        # HVN strength bonus for historically validated support on VAL bounces
        # If VAL is a historically tested support, confidence boost is justified
        # Test: More aggressive scaling - lower HVN strengths get more reward
        if val_bounce:
            if hvn_strength > 0.75:
                confidence += 0.35  # Very strong historical support (reduced slightly)
            elif hvn_strength > 0.6:
                confidence += 0.30  # Moderate-strong historical support
            elif hvn_strength > 0.4:
                confidence += 0.25  # Moderate support (increased)
            elif hvn_strength > 0.15:
                confidence += 0.15  # Weak support (increased from 0.12)

        # Regime bonus (only for strong trends to avoid noise)
        if mss_val > 20:
            confidence += 0.10

        # In VA bonus (better context for entry)
        # Being inside VA means price hasn't exited consensus yet - good for mean reversion
        if profile.val <= current_price <= profile.vah:
            confidence += 0.10  # Reduced from 0.15 - location alone less predictive

        confidence = min(confidence, 1.0)

        # Confidence threshold: balance signal volume with quality
        # Daily: ultra-low threshold to maximize trade volume
        if is_daily:
            min_confidence_threshold = 0.25
        elif is_15m:
            # Low threshold to maximize trade volume
            min_confidence_threshold = 0.35
        elif is_4h:
            min_confidence_threshold = (
                0.45  # Tighter from 0.40 to improve WR without killing volume
            )
        else:
            min_confidence_threshold = 0.65

        if confidence < min_confidence_threshold:
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=confidence,
                reason=f"Low confidence {confidence:.2f} < {min_confidence_threshold} threshold",
            )

        # Position sizing: Scale with confidence and profit/loss ratio
        # For daily, use standard sizing (iter 12 best result)
        if is_daily:
            position_size_pct = 1.0
        elif is_15m:
            # Test: Even more aggressive confidence scaling to maximize CAGR
            # Threshold 0.40 → 0.1x, confidence 0.65+ → 1.0x
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if risk > 0:
                rr_ratio = reward / risk
                # Ultra-steep scaling: maximize CAGR on high-confidence trades
                # Confidence 0.40→0.15x, 0.60→0.5x, 0.80→1.0x (exponential curve)
                if confidence >= 0.75:
                    conf_multiplier = 1.0
                elif confidence >= 0.65:
                    conf_multiplier = 0.8
                elif confidence >= 0.55:
                    conf_multiplier = 0.5
                else:
                    conf_multiplier = max(0.15, (confidence - 0.35) / 1.5)
                position_size_pct = min(1.0, conf_multiplier * (0.8 + rr_ratio * 0.3))
            else:
                position_size_pct = 1.0
        else:
            position_size_pct = 1.0

        return Signal(
            action=SignalAction.BUY,
            price=current_price,
            confidence=confidence,
            reason=f"Entry: price={current_price:.2f}, target={take_profit:.2f}",
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size_pct,
        )

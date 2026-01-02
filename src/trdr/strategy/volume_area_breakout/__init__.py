"""VolumeAreaBreakout strategy."""

from .strategy import (
    VolumeAreaBreakoutConfig,
    VolumeAreaBreakoutStrategy,
    calculate_atr,
    calculate_volume_profile,
)

# Re-export generate_signal as standalone function for backwards compatibility
# TODO: Update main.py to use strategy.generate_signal() instead
def generate_volume_area_breakout_signal(bars, position, atr_threshold=2.0, stop_loss_multiplier=1.75):
    """Backwards-compatible wrapper."""
    from ..types import Signal, SignalAction
    strategy = VolumeAreaBreakoutStrategy(
        VolumeAreaBreakoutConfig(
            symbol="",
            timeframe="",
            atr_threshold=atr_threshold,
            stop_loss_multiplier=stop_loss_multiplier,
        )
    )
    return strategy.generate_signal(bars, position)

__all__ = [
    "VolumeAreaBreakoutConfig",
    "VolumeAreaBreakoutStrategy",
    "calculate_atr",
    "calculate_volume_profile",
    "generate_volume_area_breakout_signal",
]

"""Trading calendar for market hours filtering."""

from datetime import datetime

# US market holidays (fixed dates - doesn't handle observed days)
US_HOLIDAYS = {
    # New Year's Day
    (1, 1),
    # MLK Day - 3rd Monday of January (approx)
    # Presidents Day - 3rd Monday of February (approx)
    # Good Friday - varies
    # Memorial Day - last Monday of May (approx)
    # Juneteenth
    (6, 19),
    # Independence Day
    (7, 4),
    # Labor Day - 1st Monday of September (approx)
    # Thanksgiving - 4th Thursday of November (approx)
    # Christmas
    (12, 25),
}


def is_trading_day(timestamp: str, asset_type: str) -> bool:
    """Check if timestamp falls on a trading day.

    Args:
        timestamp: ISO timestamp
        asset_type: "crypto" (24/7) or "stock" (M-F, exclude holidays)

    Returns:
        True if trading is allowed
    """
    if asset_type == "crypto":
        return True

    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    # Skip weekends (Saturday=5, Sunday=6)
    if dt.weekday() >= 5:
        return False

    # Skip major US holidays (simplified check)
    if (dt.month, dt.day) in US_HOLIDAYS:
        return False

    return True


def is_market_hours(timestamp: str, asset_type: str) -> bool:
    """Check if timestamp falls within market hours.

    Args:
        timestamp: ISO timestamp
        asset_type: "crypto" (24/7) or "stock" (9:30-16:00 ET)

    Returns:
        True if within market hours
    """
    if asset_type == "crypto":
        return True

    if not is_trading_day(timestamp, asset_type):
        return False

    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    # Convert to ET (simplified - assumes UTC-5, doesn't handle DST)
    # For backtesting purposes, this is usually sufficient
    et_hour = (dt.hour - 5) % 24

    # NYSE hours: 9:30 AM - 4:00 PM ET
    if et_hour < 9 or et_hour >= 16:
        return False
    if et_hour == 9 and dt.minute < 30:
        return False

    return True


def get_trading_days_in_year(asset_type: str) -> int:
    """Get approximate trading days per year.

    Args:
        asset_type: "crypto" or "stock"

    Returns:
        Trading days per year
    """
    if asset_type == "crypto":
        return 365
    return 252  # Standard stock market trading days


def filter_trading_bars(bars: list, asset_type: str) -> list:
    """Filter bars to only trading days.

    Args:
        bars: List of Bar objects with timestamp attribute
        asset_type: "crypto" or "stock"

    Returns:
        Filtered list of bars on trading days
    """
    if asset_type == "crypto":
        return bars

    return [bar for bar in bars if is_trading_day(bar.timestamp, asset_type)]

"""Multi-feed alignment for backtesting."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.market import Bar


def align_feeds(
    primary_bars: list["Bar"],
    informative_bars: list["Bar"],
) -> list["Bar | None"]:
    """Align informative feed to primary feed timestamps via forward-fill.

    For each primary bar, finds the most recent informative bar that
    precedes it. Works for both multi-timeframe (same symbol) and
    multi-symbol scenarios.

    Args:
        primary_bars: Primary feed bars (determines output length)
        informative_bars: Informative feed bars to align

    Returns:
        List of aligned bars (same length as primary_bars), with None
        where no informative bar is available yet
    """
    if not informative_bars:
        return [None] * len(primary_bars)

    aligned: list["Bar | None"] = []
    info_idx = 0

    for bar in primary_bars:
        # Advance to latest informative bar that precedes this bar
        while (
            info_idx < len(informative_bars) - 1
            and informative_bars[info_idx + 1].timestamp <= bar.timestamp
        ):
            info_idx += 1

        # Only include if informative bar timestamp <= primary bar timestamp
        if informative_bars[info_idx].timestamp <= bar.timestamp:
            aligned.append(informative_bars[info_idx])
        else:
            aligned.append(None)

    return aligned

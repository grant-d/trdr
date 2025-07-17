"""
Utilities for generating consistent filenames for market data storage.

This module provides functions to generate standardized filenames based on
trading symbols, timeframes, and data types. It ensures consistent naming
across the application for easy data organization and retrieval.
"""

import os
from typing import Optional


def generate_filename(
    symbol: str, timeframe: str, domain: Optional[str] = None, format: str = "csv"
) -> str:
    """
    Generate a filename based on symbol, timeframe, and optional domain.

    Args:
        symbol: Trading symbol (e.g., "BTC/USD", "AAPL")
        timeframe: Time interval (e.g., "1m", "1d")
        domain: Optional domain descriptor (e.g., "bars", "dollar-bars", "config")
        format: File format (e.g., "csv", "json")

    Returns:
        Generated filename string

    Examples:
        - generate_filename("BTC/USD", "1d", "bars") -> "btc_usd_1d.bars.csv"
        - generate_filename("AAPL", "1m", "dollar-bars") -> "aapl_1m.dollar-bars.csv"
        - generate_filename("MSFT", "1d", "config", "json") -> "msft_1d.config.json"
    """
    # Clean symbol - replace special characters with underscore
    clean_symbol = symbol.replace("/", "_").replace("-", "_").lower()

    # Build filename parts
    base_parts = [clean_symbol, timeframe.lower()]
    base_name = "_".join(base_parts)

    # Add domain with dot separator if provided
    if domain:
        filename = f"{base_name}.{domain.lower()}.{format}"
    else:
        filename = f"{base_name}.{format}"

    return filename


def get_data_path(filename: str, data_dir: str = "data") -> str:
    """
    Get full path for data file, creating directory if needed.

    Args:
        filename: Filename to use
        data_dir: Directory for data files (default: "data")

    Returns:
        Full path to the file
    """
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)


def generate_processed_filename(
    symbol: str, timeframe: str, bar_type: str, threshold: Optional[float] = None
) -> str:
    """
    Generate filename for processed data files.

    Args:
        symbol: Trading symbol
        timeframe: Original timeframe
        bar_type: Type of bars (cleaned, dollar, volume, tick)
        threshold: Threshold value for alternative bars

    Returns:
        Full path to output file
    """
    if threshold is not None:
        # Format threshold for filename (remove decimals for large numbers)
        if threshold >= 1000000:
            threshold_str = f"{int(threshold / 1000000)}m"
        elif threshold >= 1000:
            threshold_str = f"{int(threshold / 1000)}k"
        else:
            threshold_str = str(int(threshold))

        # Include threshold in filename
        domain = f"{bar_type}-{threshold_str}"
    else:
        domain = bar_type

    filename = generate_filename(symbol, timeframe, domain, "csv")
    return get_data_path(filename)

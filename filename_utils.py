"""
Utilities for generating consistent filenames for market data storage.

This module provides functions to generate standardized filenames based on
trading symbols, timeframes, and data types. It ensures consistent naming
across the application for easy data organization and retrieval.
"""

import os
from typing import Optional


def generate_filename(
    symbol: str,
    timeframe: str,
    domain: Optional[str] = None,
    format: str = "csv"
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
        - generate_filename("BTC/USD", "1d", "bars") -> "btc_usd_1d_bars.csv"
        - generate_filename("AAPL", "1m", "dollar-bars") -> "aapl_1m_dollar-bars.csv"
        - generate_filename("MSFT", "1d", "config", "json") -> "msft_1d_config.json"
    """
    # Clean symbol - replace special characters with underscore
    clean_symbol = symbol.replace("/", "_").replace("-", "_").lower()

    # Build filename parts
    parts = [clean_symbol, timeframe.lower()]

    if domain:
        parts.append(domain.lower())

    # Join parts and add extension
    filename = "_".join(parts) + f".{format}"

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

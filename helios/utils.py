import numpy as np
from pathlib import Path
from typing import Union


def generate_filename(symbol: str, 
                          timeframe: Union[str, int], 
                          postfix: str = "",
                          extension: str = ".csv",
                          directory: str = "./data") -> Path:
    """
    Generate standardized filenames across all Helios components.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'BTC/USD', 'AAPL')
    timeframe : Union[str, int]
        Timeframe specification:
        - For time bars: int (minutes) or str ('1min', '5min', '1hour')
        - For dollar bars: 'dollar_1000', 'dollar_5000'
        - For custom: any string identifier
    postfix : str, optional
        Additional identifier (e.g., 'fitness_history', 'cache', 'results')
    extension : str, optional
        File extension including dot (default: '.csv')
    directory : str, optional
        Base directory path (default: './data')
    
    Returns:
    --------
    Path
        Complete file path
        
    Examples:
    ---------
    >>> generate_cache_filename('BTC/USD', 1, 'cache')
    Path('./data/BTC_USD_1min_cache.csv')
    
    >>> generate_cache_filename('BTC/USD', 'dollar_1000', 'fitness_history', '.json')  
    Path('./data/BTC_USD_dollar_1000_fitness_history.json')
    
    >>> generate_cache_filename('AAPL', '1hour', 'optimization_results', '.json')
    Path('./data/AAPL_1hour_optimization_results.json')
    """
    # Clean symbol for safe filename
    safe_symbol = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
    
    # Standardize timeframe format
    if isinstance(timeframe, int):
        # Convert integer minutes to standard format
        if timeframe < 60:
            tf_str = f"{timeframe}min"
        elif timeframe < 1440:
            hours = timeframe // 60
            tf_str = f"{hours}hour" if timeframe % 60 == 0 else f"{timeframe}min"
        else:
            days = timeframe // 1440
            tf_str = f"{days}day" if timeframe % 1440 == 0 else f"{timeframe}min"
    else:
        # String timeframe - use as-is but clean for filename
        tf_str = str(timeframe).replace("/", "_").replace("\\", "_").replace(":", "_")
    
    # Build filename parts
    parts = [safe_symbol, tf_str]
    if postfix:
        parts.append(postfix)
    
    filename = "_".join(parts) + extension
    return Path(directory) / filename


def safe_prod(series) -> float:
    """
    Robustly compute the product of a pandas/numpy series, always returning a float.
    Handles numpy scalars, native types, and complex results.
    """
    prod = series.prod()
    if isinstance(prod, complex):
        prod = prod.real
    if isinstance(prod, np.generic):
        prod = prod.item()
    return float(prod)

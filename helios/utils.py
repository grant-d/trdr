import numpy as np

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

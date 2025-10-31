"""
Black-Scholes Options Pricing Model

Used to calculate theoretical prices for European-style options
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple


class BlackScholes:
    """Black-Scholes option pricing model"""

    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0:
            # Option expired
            return 0, 0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate call option price

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Call option price
        """
        if T <= 0:
            return max(0, S - K)

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option price

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Put option price
        """
        if T <= 0:
            return max(0, K - S)

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put

    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option delta

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Put delta
        """
        if T <= 0:
            return -1.0 if S < K else 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option gamma (same for calls and puts)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Option gamma
        """
        if T <= 0:
            return 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option vega (same for calls and puts)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Option vega (per 1% change in volatility)
        """
        if T <= 0:
            return 0.0

        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% change

    @staticmethod
    def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option theta (time decay per day)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Put theta (per day)
        """
        if T <= 0:
            return 0.0

        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r, sigma)

        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2))

        return theta / 365  # Convert to daily theta


def calculate_historical_volatility(prices: np.ndarray, window: int = 30) -> float:
    """
    Calculate historical volatility from price series

    Args:
        prices: Array of prices
        window: Lookback window for volatility calculation

    Returns:
        Annualized volatility
    """
    if len(prices) < 2:
        return 0.3  # Default volatility

    returns = np.log(prices[1:] / prices[:-1])
    recent_returns = returns[-window:] if len(returns) > window else returns

    volatility = np.std(recent_returns) * np.sqrt(252)  # Annualize
    return volatility


if __name__ == '__main__':
    # Test Black-Scholes pricing
    S = 175.00  # AAPL price
    K = 170.00  # Strike price
    T = 30 / 365  # 30 days to expiration
    r = 0.05  # 5% risk-free rate
    sigma = 0.30  # 30% volatility

    bs = BlackScholes()
    put_px = bs.put_price(S, K, T, r, sigma)
    delta = bs.put_delta(S, K, T, r, sigma)

    print(f"Put Price: ${put_px:.2f}")
    print(f"Put Delta: {delta:.4f}")

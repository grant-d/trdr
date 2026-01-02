"""Tests for trading strategy.

Your strategy must achieve:
- Sortino ratio > 1.5
- Max drawdown < 15%
- Win rate > 50%
- Profit factor > 1.5
"""

import numpy as np
import pytest
from strategy import generate_signals


def generate_price_data(seed: int = 42, n: int = 1000) -> np.ndarray:
    """Generate synthetic price data with exploitable patterns.

    The data has:
    - Trending periods (momentum works)
    - Mean-reverting periods (fade works)
    - Regime changes
    """
    np.random.seed(seed)

    prices = [100.0]
    regime = "trend"
    regime_length = 0
    trend_dir = 1

    for i in range(n - 1):
        regime_length += 1

        # Regime switching
        if regime_length > np.random.randint(30, 100):
            regime = np.random.choice(["trend", "mean_revert", "choppy"])
            regime_length = 0
            trend_dir = np.random.choice([-1, 1])

        if regime == "trend":
            # Trending: momentum continues
            drift = 0.0005 * trend_dir
            noise = np.random.normal(0, 0.008)
            ret = drift + noise
        elif regime == "mean_revert":
            # Mean reverting: fade moves
            last_ret = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            drift = -0.3 * last_ret  # Mean revert
            noise = np.random.normal(0, 0.012)
            ret = drift + noise
        else:
            # Choppy: random walk
            ret = np.random.normal(0, 0.015)

        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Floor at 1.0

    return np.array(prices)


def compute_returns(prices: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """Compute strategy returns.

    Return[i] = signal[i] * (price[i+1] - price[i]) / price[i]
    """
    price_returns = np.diff(prices) / prices[:-1]
    # Signal at i determines position for return from i to i+1
    strategy_returns = signals[:-1] * price_returns
    return strategy_returns


def sortino_ratio(returns: np.ndarray, target: float = 0.0) -> float:
    """Compute annualized Sortino ratio."""
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return np.inf if np.mean(returns) > 0 else 0.0
    downside_std = np.std(downside)
    # Annualize assuming daily returns
    return np.mean(returns) / downside_std * np.sqrt(252)


def max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from returns."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    return np.max(drawdowns)


def win_rate(returns: np.ndarray) -> float:
    """Compute win rate (fraction of positive returns)."""
    non_zero = returns[returns != 0]
    if len(non_zero) == 0:
        return 0.0
    return np.mean(non_zero > 0)


def profit_factor(returns: np.ndarray) -> float:
    """Compute profit factor (gross profits / gross losses)."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return gains / losses


class TestStrategyPerformance:
    """Performance tests for the trading strategy."""

    @pytest.fixture
    def price_data(self):
        return generate_price_data(seed=42, n=1000)

    @pytest.fixture
    def strategy_returns(self, price_data):
        signals = generate_signals(price_data)
        return compute_returns(price_data, signals)

    def test_sortino_ratio(self, strategy_returns):
        """Strategy must achieve Sortino ratio > 1.5"""
        ratio = sortino_ratio(strategy_returns)
        assert ratio > 1.5, f"Sortino ratio {ratio:.2f} < 1.5"

    def test_max_drawdown(self, strategy_returns):
        """Strategy must have max drawdown < 15%"""
        dd = max_drawdown(strategy_returns)
        assert dd < 0.15, f"Max drawdown {dd:.1%} > 15%"

    def test_win_rate(self, strategy_returns):
        """Strategy must have win rate > 50%"""
        wr = win_rate(strategy_returns)
        assert wr > 0.50, f"Win rate {wr:.1%} < 50%"

    def test_profit_factor(self, strategy_returns):
        """Strategy must have profit factor > 1.5"""
        pf = profit_factor(strategy_returns)
        assert pf > 1.5, f"Profit factor {pf:.2f} < 1.5"

    def test_total_return_positive(self, strategy_returns):
        """Strategy must have positive total return"""
        total = np.prod(1 + strategy_returns) - 1
        assert total > 0, f"Total return {total:.1%} is negative"


class TestStrategyRobustness:
    """Test strategy on different data seeds."""

    @pytest.mark.parametrize("seed", [123, 456, 789])
    def test_positive_sortino_different_seeds(self, seed):
        """Strategy should have positive Sortino on different data."""
        prices = generate_price_data(seed=seed, n=1000)
        signals = generate_signals(prices)
        returns = compute_returns(prices, signals)
        ratio = sortino_ratio(returns)
        assert ratio > 0.5, f"Sortino {ratio:.2f} < 0.5 on seed {seed}"


class TestStrategyConstraints:
    """Test that strategy follows constraints."""

    def test_signals_in_range(self):
        """Signals must be in [-1, 1]"""
        prices = generate_price_data()
        signals = generate_signals(prices)
        assert np.all(signals >= -1.0), "Signal below -1"
        assert np.all(signals <= 1.0), "Signal above 1"

    def test_signals_length(self):
        """Signals array must match prices length"""
        prices = generate_price_data()
        signals = generate_signals(prices)
        assert len(signals) == len(prices)


def print_strategy_stats():
    """Helper to print strategy statistics."""
    prices = generate_price_data()
    signals = generate_signals(prices)
    returns = compute_returns(prices, signals)

    print(f"Total return: {np.prod(1 + returns) - 1:.1%}")
    print(f"Sortino ratio: {sortino_ratio(returns):.2f}")
    print(f"Max drawdown: {max_drawdown(returns):.1%}")
    print(f"Win rate: {win_rate(returns):.1%}")
    print(f"Profit factor: {profit_factor(returns):.2f}")
    print(f"Num trades: {np.sum(np.diff(signals) != 0)}")


if __name__ == "__main__":
    print_strategy_stats()

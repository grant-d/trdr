"""
Performance evaluation functions for Helios Trader
Includes Sortino Ratio, Calmar Ratio, and other metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def calculate_geometric_mean(returns: pd.Series) -> float:
    """
    Calculate geometric mean of returns

    Parameters:
    -----------
    returns : pd.Series
        Series of returns (as decimals, not percentages)

    Returns:
    --------
    float
        Geometric mean return
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    # Convert returns to growth factors (1 + r)
    growth_factors = 1 + returns

    # Calculate geometric mean: (product of growth factors)^(1/n) - 1
    # Use pandas prod() method for better type handling
    product = float(growth_factors.prod())
    geometric_mean = product ** (1 / len(returns)) - 1

    return float(geometric_mean)


def calculate_returns_metrics(
    returns: pd.Series, risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate comprehensive return metrics

    Parameters:
    -----------
    returns : pd.Series
        Series of returns (as decimals, not percentages)
    risk_free_rate : float
        Annual risk-free rate

    Returns:
    --------
    Dict[str, float]
        Dictionary of performance metrics
    """
    # Ensure we have valid returns
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            "total_return": 0,
            "annualized_return": 0,
            "annualized_return_geometric": 0,
            "arithmetic_mean_return": 0,
            "geometric_mean_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
        }

    # Basic metrics
    # Calculate total return using pandas for consistent types
    total_return = float((1.0 + returns).prod() - 1.0)

    # Calculate both arithmetic and geometric means
    arithmetic_mean = float(returns.mean())
    geometric_mean = calculate_geometric_mean(returns)

    # Annualized metrics (assuming daily returns)
    periods_per_year = 252
    n_periods = len(returns)

    # Traditional annualized return (using total return)
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # Geometric annualized return (more accurate for compounding)
    annualized_return_geometric = (1 + geometric_mean) ** periods_per_year - 1

    annualized_vol = float(returns.std()) * np.sqrt(periods_per_year)

    # Sharpe Ratio
    excess_returns = returns - risk_free_rate / periods_per_year
    returns_std = float(returns.std())
    excess_mean = float(excess_returns.mean())
    sharpe_ratio = (
        excess_mean / returns_std * np.sqrt(periods_per_year)
        if returns_std > 0
        else 0.0
    )

    # Sortino Ratio
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)

    # Maximum Drawdown and Calmar Ratio
    cumulative_returns = (1 + returns).cumprod()
    max_drawdown = calculate_max_drawdown(cumulative_returns)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Win rate
    returns_array = np.asarray(returns.values)
    win_rate = float(np.mean(returns_array > 0.0)) * 100

    return {
        "total_return": total_return * 100,  # As percentage
        "annualized_return": annualized_return * 100,
        "annualized_return_geometric": annualized_return_geometric * 100,
        "arithmetic_mean_return": arithmetic_mean * 100,
        "geometric_mean_return": geometric_mean * 100,
        "volatility": annualized_vol * 100,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown * 100,
        "win_rate": win_rate,
    }


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio

    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year

    Returns:
    --------
    float
        Sortino ratio
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0

    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year

    # Excess returns
    excess_returns = returns - daily_rf

    # Downside returns (only negative excess returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # No downside, return a high value (capped)
        return 10.0

    # Downside deviation
    downside_std = np.sqrt(np.mean(downside_returns**2))

    if downside_std == 0:
        return 10.0  # Cap at 10 if no downside deviation

    # Annualized Sortino Ratio
    sortino = (
        float(excess_returns.mean()) / downside_std * np.sqrt(periods_per_year)
    )

    return sortino


def calculate_sortino_ratio_classic(
    returns: pd.Series, mar: float = 0.0, periods_per_year: int = 365
) -> float:
    """
    Calculate Sortino Ratio using classic approach (MAR-based, no risk-free rate)
    This matches the old notebook implementation

    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    mar : float
        Minimum Acceptable Return (default 0)
    periods_per_year : int
        Number of periods per year (365 for dollar bars)

    Returns:
    --------
    float
        Sortino ratio
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0

    # Downside returns (below MAR)
    downside_returns = returns[returns < mar]

    if len(downside_returns) == 0:
        # No downside returns
        return float("inf")

    # Downside deviation
    downside_deviation = downside_returns.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        return float("inf")

    # Mean return
    mean_return = returns.mean()

    # Annualized Sortino Ratio
    sortino = (mean_return - mar) * np.sqrt(periods_per_year) / downside_deviation

    return float(sortino)


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown

    Parameters:
    -----------
    cumulative_returns : pd.Series
        Cumulative returns series (starting from 1)

    Returns:
    --------
    float
        Maximum drawdown (negative value)
    """
    if len(cumulative_returns) == 0:
        return 0

    # Calculate running maximum
    running_max = cumulative_returns.expanding().max()

    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max

    # Return maximum drawdown (most negative value)
    return float(drawdown.min())


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar Ratio

    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Number of periods per year

    Returns:
    --------
    float
        Calmar ratio
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0

    # Calculate annualized return
    total_return = float((1.0 + returns).prod() - 1.0)
    n_periods = len(returns)
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    max_dd = calculate_max_drawdown(cumulative_returns)

    if max_dd == 0:
        return 0

    return float(annualized_return / abs(max_dd))


def evaluate_strategy_performance(
    results_df: pd.DataFrame, initial_capital: float = 100000
) -> Dict[str, float]:
    """
    Evaluate trading strategy performance

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with trading results (must have 'portfolio_value' column)
    initial_capital : float
        Initial capital

    Returns:
    --------
    Dict[str, float]
        Performance metrics
    """
    if "portfolio_value" not in results_df.columns:
        raise ValueError("results_df must have 'portfolio_value' column")

    # Calculate returns
    portfolio_values = results_df["portfolio_value"]
    returns = portfolio_values.pct_change().dropna()

    # Get all metrics
    metrics = calculate_returns_metrics(returns)

    # Add strategy-specific metrics
    final_value = float(portfolio_values.values[-1])
    metrics["final_portfolio_value"] = final_value
    metrics["total_pnl"] = final_value - initial_capital
    metrics["total_return_pct"] = (final_value / initial_capital - 1) * 100

    return metrics


def compare_strategies(
    strategy_results: Dict[str, pd.DataFrame], initial_capital: float = 100000
) -> pd.DataFrame:
    """
    Compare multiple trading strategies

    Parameters:
    -----------
    strategy_results : Dict[str, pd.DataFrame]
        Dictionary of strategy names to results DataFrames
    initial_capital : float
        Initial capital for all strategies

    Returns:
    --------
    pd.DataFrame
        Comparison of all strategies
    """
    comparison = {}

    for name, results in strategy_results.items():
        metrics = evaluate_strategy_performance(results, initial_capital)
        comparison[name] = metrics

    return pd.DataFrame(comparison).T


def calculate_rolling_metrics(
    results_df: pd.DataFrame, window: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with trading results
    window : int
        Rolling window size (default 252 for 1 year)

    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling metrics
    """
    if "portfolio_value" not in results_df.columns:
        raise ValueError("results_df must have 'portfolio_value' column")

    # Calculate returns
    returns = results_df["portfolio_value"].pct_change()

    # Initialize rolling metrics DataFrame
    rolling_metrics = pd.DataFrame(index=results_df.index)

    # Rolling returns
    rolling_metrics["rolling_return"] = (
        returns.rolling(window).apply(lambda x: (1 + x).prod() - 1) * 100
    )

    # Rolling volatility
    rolling_metrics["rolling_volatility"] = (
        returns.rolling(window).std() * np.sqrt(252) * 100
    )

    # Rolling Sharpe
    rolling_metrics["rolling_sharpe"] = returns.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )

    # Rolling max drawdown
    def rolling_max_dd(values):
        cumsum = (1 + values).cumprod()
        return calculate_max_drawdown(cumsum) * 100

    rolling_metrics["rolling_max_drawdown"] = returns.rolling(window).apply(
        rolling_max_dd
    )

    return rolling_metrics


def generate_performance_report(
    results_df: pd.DataFrame, trades_summary: Dict, initial_capital: float = 100000
) -> str:
    """
    Generate a comprehensive performance report

    Parameters:
    -----------
    results_df : pd.DataFrame
        Trading results DataFrame
    trades_summary : Dict
        Trade summary statistics
    initial_capital : float
        Initial capital

    Returns:
    --------
    str
        Formatted performance report
    """
    # Calculate performance metrics
    from chalk import green, red, yellow, cyan, bold, black
    
    metrics = evaluate_strategy_performance(results_df, initial_capital)

    report = []
    report.append(bold("=" * 60))
    report.append(bold(cyan("HELIOS TRADER PERFORMANCE REPORT")))
    report.append(bold("=" * 60))

    # Overall Performance
    report.append(f"\n{bold('OVERALL PERFORMANCE')}")
    report.append(black("-" * 30))
    report.append(f"Initial Capital:        {black('$')}{initial_capital:>13,.2f}")
    
    # Color code based on profit/loss
    final_val = metrics['final_portfolio_value']
    pnl = metrics['total_pnl']
    total_ret = metrics['total_return_pct']
    ann_ret = metrics['annualized_return']
    
    if pnl > 0:
        report.append(f"Final Portfolio Value:  {green(f'${final_val:>13,.2f}')}")
        report.append(f"Total P&L:              {green(f'+${pnl:>12,.2f}')}")
        report.append(f"Total Return:           {green(f'+{total_ret:>12.2f}%')}")
        report.append(f"Annualized Return:      {green(f'{ann_ret:>13.2f}%')}")
    else:
        report.append(f"Final Portfolio Value:  {red(f'${final_val:>13,.2f}')}")
        report.append(f"Total P&L:              {red(f'${pnl:>13,.2f}')}")
        report.append(f"Total Return:           {red(f'{total_ret:>13.2f}%')}")
        report.append(f"Annualized Return:      {red(f'{ann_ret:>13.2f}%')}")
    
    report.append(
        f"Geometric Ann. Return:  {green(f'{metrics.get('annualized_return_geometric', ann_ret):>13.2f}%') if ann_ret > 0 else red(f'{metrics.get('annualized_return_geometric', ann_ret):>13.2f}%')}"
    )

    # Risk Metrics
    report.append(f"\n{bold('RISK METRICS')}")
    report.append(black("-" * 30))
    
    # Volatility - lower is better
    vol = metrics['volatility']
    vol_color = green if vol < 20 else yellow if vol < 40 else red
    report.append(f"Volatility (Annual):    {vol_color(f'{vol:>13.2f}%')}")
    
    # Max Drawdown - closer to 0 is better
    dd = metrics['max_drawdown']
    dd_color = green if dd > -5 else yellow if dd > -15 else red
    report.append(f"Maximum Drawdown:       {dd_color(f'{dd:>13.2f}%')}")
    
    # Ratios - higher is better
    sharpe = metrics['sharpe_ratio']
    sortino = metrics['sortino_ratio']
    calmar = metrics['calmar_ratio']
    
    report.append(f"Sharpe Ratio:           {green(f'{sharpe:>13.2f}') if sharpe > 1 else yellow(f'{sharpe:>13.2f}') if sharpe > 0.5 else red(f'{sharpe:>13.2f}')}")
    report.append(f"Sortino Ratio:          {green(f'{sortino:>13.2f}') if sortino > 2 else yellow(f'{sortino:>13.2f}') if sortino > 1 else red(f'{sortino:>13.2f}')}")
    report.append(f"Calmar Ratio:           {green(f'{calmar:>13.2f}') if calmar > 3 else yellow(f'{calmar:>13.2f}') if calmar > 1 else red(f'{calmar:>13.2f}')}")

    # Mean Returns (for advanced analysis)
    report.append(f"\n{bold('RETURN STATISTICS')}")
    report.append(black("-" * 30))
    report.append(
        f"Arithmetic Mean:        {black(f'{metrics.get('arithmetic_mean_return', 0):>13.4f}%')}"
    )
    report.append(
        f"Geometric Mean:         {black(f'{metrics.get('geometric_mean_return', 0):>13.4f}%')}"
    )

    # Trading Statistics
    report.append(f"\n{bold('TRADING STATISTICS')}")
    report.append(black("-" * 30))
    report.append(f"Total Trades:           {cyan(str(trades_summary.get('total_trades', 0)).rjust(14))}")
    report.append(f"Position Cycles:        {str(trades_summary.get('position_cycles', 0)).rjust(14)}")
    
    wins = trades_summary.get('winning_trades', 0)
    losses = trades_summary.get('losing_trades', 0)
    report.append(f"Winning Trades:         {green(str(wins).rjust(14))}")
    report.append(f"Losing Trades:          {red(str(losses).rjust(14))}")
    
    # Win Rate - color based on performance
    win_rate = trades_summary.get('win_rate', 0)
    wr_color = green if win_rate > 60 else yellow if win_rate > 40 else red
    report.append(f"Win Rate:               {wr_color(f'{win_rate:>13.2f}%')}")
    
    # Average win/loss
    avg_win = trades_summary.get('avg_win', 0)
    avg_loss = trades_summary.get('avg_loss', 0)
    report.append(f"Average Win:            {green(f'+{avg_win:>12.2f}%')}")
    report.append(f"Average Loss:           {red(f'{avg_loss:>13.2f}%')}")
    
    # Profit Factor - higher is better
    pf = trades_summary.get('profit_factor', 0)
    pf_color = green if pf > 2 else yellow if pf > 1.5 else red
    report.append(f"Profit Factor:          {pf_color(f'{pf:>13.2f}')}")

    report.append("\n" + bold("=" * 60))

    return "\n".join(report)

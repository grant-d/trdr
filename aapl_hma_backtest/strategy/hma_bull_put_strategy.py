"""
HMA Bull Put Spread Strategy

Entry Conditions:
- HMA 50 is trending up
- HMA 200 is trending up

Position:
- Sell put option just below current price (short put)
- Buy put option further below for protection (long put)
- This creates a bull put spread (credit spread)

Exit Conditions:
- Take profit at 50% of max profit
- Stop loss at 200% of credit received
- Close at expiration
- Exit if HMA trend reverses
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from options.black_scholes import BlackScholes, calculate_historical_volatility
from options.position import OptionPosition, BullPutSpread, Portfolio


class HMABullPutStrategy:
    """HMA-based Bull Put Spread Strategy"""

    def __init__(self,
                 initial_capital: float = 100000,
                 short_put_delta: float = -0.30,  # Sell put at ~30 delta
                 spread_width: float = 5.0,  # $5 spread width
                 days_to_expiration: int = 30,  # 30 DTE options
                 profit_target: float = 0.50,  # Take profit at 50% of max profit
                 stop_loss: float = 2.0,  # Stop loss at 2x credit received
                 risk_free_rate: float = 0.05,  # 5% risk-free rate
                 max_positions: int = 5):  # Maximum concurrent positions
        """
        Initialize strategy

        Args:
            initial_capital: Starting capital
            short_put_delta: Target delta for short put
            spread_width: Width between short and long puts
            days_to_expiration: DTE for new positions
            profit_target: Profit target as % of max profit
            stop_loss: Stop loss as multiple of credit received
            risk_free_rate: Risk-free rate for pricing
            max_positions: Maximum number of concurrent positions
        """
        self.portfolio = Portfolio(initial_capital)
        self.short_put_delta = short_put_delta
        self.spread_width = spread_width
        self.days_to_expiration = days_to_expiration
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.risk_free_rate = risk_free_rate
        self.max_positions = max_positions

    def find_strike_by_delta(self, spot_price: float, target_delta: float,
                            time_to_expiration: float, volatility: float) -> float:
        """
        Find strike price that gives target delta for a put option

        Args:
            spot_price: Current stock price
            target_delta: Target delta (negative for puts)
            time_to_expiration: Time to expiration in years
            volatility: Implied volatility

        Returns:
            Strike price
        """
        bs = BlackScholes()

        # Search for strike using binary search
        lower_strike = spot_price * 0.70  # Start at 70% of spot
        upper_strike = spot_price * 0.99  # End just below spot

        for _ in range(50):  # Maximum iterations
            mid_strike = (lower_strike + upper_strike) / 2
            delta = bs.put_delta(spot_price, mid_strike, time_to_expiration,
                                self.risk_free_rate, volatility)

            if abs(delta - target_delta) < 0.01:
                return round(mid_strike, 2)

            if delta < target_delta:  # Delta too low (option too far OTM)
                lower_strike = mid_strike
            else:  # Delta too high (option too close to ATM)
                upper_strike = mid_strike

        return round(mid_strike, 2)

    def check_entry_signal(self, row: pd.Series) -> bool:
        """
        Check if entry conditions are met

        Args:
            row: DataFrame row with OHLC and indicators

        Returns:
            True if entry signal
        """
        # Both HMA 50 and HMA 200 must be trending up
        return row['HMA_50_UP'] and row['HMA_200_UP']

    def check_exit_signal(self, row: pd.Series) -> bool:
        """
        Check if exit conditions are met (trend reversal)

        Args:
            row: DataFrame row with OHLC and indicators

        Returns:
            True if exit signal
        """
        # Exit if either HMA turns down
        return not (row['HMA_50_UP'] and row['HMA_200_UP'])

    def create_bull_put_spread(self, current_date: datetime, spot_price: float,
                               volatility: float) -> Optional[BullPutSpread]:
        """
        Create a bull put spread

        Args:
            current_date: Current date
            spot_price: Current stock price
            volatility: Historical volatility

        Returns:
            BullPutSpread object or None
        """
        bs = BlackScholes()
        time_to_expiration = self.days_to_expiration / 365

        # Find short put strike
        short_strike = self.find_strike_by_delta(spot_price, self.short_put_delta,
                                                 time_to_expiration, volatility)

        # Long put strike is spread_width below short strike
        long_strike = short_strike - self.spread_width

        # Calculate option prices
        expiration_date = current_date + timedelta(days=self.days_to_expiration)

        short_put_price = bs.put_price(spot_price, short_strike, time_to_expiration,
                                       self.risk_free_rate, volatility)
        long_put_price = bs.put_price(spot_price, long_strike, time_to_expiration,
                                      self.risk_free_rate, volatility)

        # Net credit
        net_credit = short_put_price - long_put_price

        if net_credit <= 0:
            return None  # Invalid spread

        # Create positions
        short_put = OptionPosition(
            option_type='put',
            strike=short_strike,
            expiration=expiration_date,
            entry_date=current_date,
            entry_price=short_put_price,
            quantity=-1,  # Negative for short
            underlying_entry_price=spot_price
        )

        long_put = OptionPosition(
            option_type='put',
            strike=long_strike,
            expiration=expiration_date,
            entry_date=current_date,
            entry_price=long_put_price,
            quantity=1,  # Positive for long
            underlying_entry_price=spot_price
        )

        spread = BullPutSpread(
            short_put=short_put,
            long_put=long_put,
            entry_date=current_date,
            net_credit=net_credit
        )

        return spread

    def calculate_spread_value(self, spread: BullPutSpread, current_date: datetime,
                              spot_price: float, volatility: float) -> Tuple[float, float]:
        """
        Calculate current value of spread

        Args:
            spread: BullPutSpread object
            current_date: Current date
            spot_price: Current stock price
            volatility: Current volatility

        Returns:
            Tuple of (short_put_price, long_put_price)
        """
        bs = BlackScholes()

        # Calculate time remaining
        days_remaining = (spread.short_put.expiration - current_date).days
        time_to_expiration = max(days_remaining / 365, 0.001)  # Avoid zero

        # If expired, use intrinsic value
        if days_remaining <= 0:
            short_value = max(0, spread.short_put.strike - spot_price)
            long_value = max(0, spread.long_put.strike - spot_price)
            return short_value, long_value

        # Calculate option prices
        short_value = bs.put_price(spot_price, spread.short_put.strike,
                                   time_to_expiration, self.risk_free_rate, volatility)
        long_value = bs.put_price(spot_price, spread.long_put.strike,
                                  time_to_expiration, self.risk_free_rate, volatility)

        return short_value, long_value

    def run_backtest(self, df: pd.DataFrame) -> Portfolio:
        """
        Run backtest on historical data

        Args:
            df: DataFrame with OHLC data and HMA indicators

        Returns:
            Portfolio object with results
        """
        position_ids = []

        for idx, row in df.iterrows():
            current_date = idx
            spot_price = row['Close']

            # Calculate volatility
            price_history = df.loc[:current_date, 'Close'].values
            volatility = calculate_historical_volatility(price_history, window=30)

            # Manage existing positions
            positions_to_close = []

            for pos_id, spread in self.portfolio.positions.items():
                # Calculate current value
                short_value, long_value = self.calculate_spread_value(
                    spread, current_date, spot_price, volatility
                )

                # Calculate P&L
                pnl = spread.calculate_pnl(short_value, long_value)
                max_profit = spread.max_profit()

                # Check exit conditions
                close_reason = None

                # 1. Expiration
                if spread.is_expired(current_date):
                    close_reason = 'expiration'

                # 2. Profit target
                elif pnl >= max_profit * self.profit_target:
                    close_reason = 'profit_target'

                # 3. Stop loss
                elif pnl <= -spread.net_credit * self.stop_loss * 100:
                    close_reason = 'stop_loss'

                # 4. Trend reversal
                elif self.check_exit_signal(row):
                    close_reason = 'trend_reversal'

                if close_reason:
                    positions_to_close.append((pos_id, short_value, long_value, close_reason))

            # Close positions
            for pos_id, short_value, long_value, reason in positions_to_close:
                self.portfolio.close_position(pos_id, current_date, short_value,
                                            long_value, reason)

            # Check for new entries
            if (len(self.portfolio.positions) < self.max_positions and
                self.check_entry_signal(row)):

                # Create new spread
                spread = self.create_bull_put_spread(current_date, spot_price, volatility)

                if spread:
                    pos_id = self.portfolio.add_position(spread)
                    position_ids.append(pos_id)

            # Calculate positions value
            positions_value = 0
            for spread in self.portfolio.positions.values():
                short_value, long_value = self.calculate_spread_value(
                    spread, current_date, spot_price, volatility
                )
                pnl = spread.calculate_pnl(short_value, long_value)
                positions_value += pnl

            # Update equity curve
            self.portfolio.update_equity(current_date, positions_value)

        # Close any remaining positions at end
        final_date = df.index[-1]
        final_price = df.iloc[-1]['Close']

        for pos_id in list(self.portfolio.positions.keys()):
            spread = self.portfolio.positions[pos_id]
            short_value, long_value = self.calculate_spread_value(
                spread, final_date, final_price, volatility
            )
            self.portfolio.close_position(pos_id, final_date, short_value,
                                        long_value, 'backtest_end')

        return self.portfolio

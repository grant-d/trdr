"""
Portfolio simulation engine for backtesting and walk-forward optimization.
Handles order management, position tracking, fees, and slippage.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import math

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side enumeration"""

    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderStatus(Enum):
    """Order status enumeration"""

    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation"""

    order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    trail_amount: Optional[float] = None  # For trailing stops (absolute)
    trail_percent: Optional[float] = None  # For trailing stops (percentage)
    time_created: Optional[datetime] = None
    time_filled: Optional[datetime] = None
    fill_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    _trail_reference_price: Optional[float] = None  # Internal tracking


@dataclass
class Position:
    """Position representation"""

    symbol: str
    quantity: float
    average_price: float
    current_price: float
    realized_pnl: float = 0.0
    commission_paid: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return (self.current_price - self.average_price) * self.quantity

    @property
    def total_pnl(self) -> float:
        """Total profit/loss including realized and unrealized"""
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class Trade:
    """Executed trade record"""

    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime


@dataclass
class Bar:
    """OHLCV bar data"""

    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


@dataclass
class PortfolioState:
    """Portfolio state snapshot"""

    timestamp: datetime
    cash: float
    positions: Dict[str, Position]
    equity: float
    margin_used: float = 0.0
    buying_power: float = 0.0


class PortfolioEngine:
    """
    Portfolio simulation engine for backtesting and paper trading.

    Parameters:
    -----------
    initial_balance : float
        Starting cash balance
    commission_rate : float
        Commission per trade (flat fee)
    commission_pct : float
        Commission as percentage of trade value
    slippage_pct : float
        Slippage as percentage of price
    slippage_fixed : float
        Fixed slippage amount
    allow_shorts : bool
        Whether short selling is allowed
    margin_requirement : float
        Margin requirement for shorts (e.g., 0.5 = 50%)
    symbol : str
        Trading symbol (for reference only)
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        commission_rate: float = 0.0,
        commission_pct: float = 0.001,  # 0.1%
        slippage_pct: float = 0.0001,  # 0.01%
        slippage_fixed: float = 0.0,
        allow_shorts: bool = False,
        margin_requirement: float = 0.5,
        symbol: str = "UNKNOWN",
    ):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.commission_rate = commission_rate
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.slippage_fixed = slippage_fixed
        self.allow_shorts = allow_shorts
        self.margin_requirement = margin_requirement
        self.symbol = symbol

        # State tracking
        self.position: Optional[Position] = None
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.order_counter = 0
        self.trade_counter = 0
        self.current_bar: Optional[Bar] = None
        self.portfolio_history: List[PortfolioState] = []

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"O{self.order_counter:06d}"

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_counter += 1
        return f"T{self.trade_counter:06d}"

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade"""
        trade_value = quantity * price
        pct_commission = trade_value * self.commission_pct
        return self.commission_rate + pct_commission

    def _calculate_slippage(self, price: float, side: OrderSide) -> float:
        """Calculate slippage for a trade"""
        pct_slippage = price * self.slippage_pct
        total_slippage = self.slippage_fixed + pct_slippage

        # Buy orders get worse (higher) prices
        if side in [OrderSide.BUY, OrderSide.COVER]:
            return total_slippage
        # Sell orders get worse (lower) prices
        else:
            return -total_slippage

    def _can_afford(self, quantity: float, price: float, side: OrderSide) -> bool:
        """Check if we have sufficient funds for the order"""
        trade_value = quantity * price
        commission = self._calculate_commission(quantity, price)

        if side == OrderSide.BUY:
            required_cash = trade_value + commission
            return self.cash >= required_cash
        elif side == OrderSide.SHORT and self.allow_shorts:
            # Check margin requirement
            margin_required = trade_value * self.margin_requirement
            return self.cash >= margin_required + commission

        return True  # SELL and COVER don't require cash

    def _has_position_to_sell(self, quantity: float) -> bool:
        """Check if we have sufficient position to sell"""
        if not self.position:
            return False
        return self.position.quantity >= quantity

    def _has_short_to_cover(self, quantity: float) -> bool:
        """Check if we have sufficient short position to cover"""
        if not self.position:
            return False
        return -self.position.quantity >= quantity

    def place_order(
        self,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Place an order

        Parameters:
        -----------
        side : OrderSide
            BUY, SELL, SHORT, or COVER
        quantity : float
            Number of shares
        order_type : OrderType
            MARKET, LIMIT, STOP, STOP_LIMIT, or TRAILING_STOP
        price : Optional[float]
            Limit price for LIMIT and STOP_LIMIT orders
        stop_price : Optional[float]
            Stop trigger price for STOP and STOP_LIMIT orders
        trail_amount : Optional[float]
            Trailing amount for TRAILING_STOP (absolute)
        trail_percent : Optional[float]
            Trailing percentage for TRAILING_STOP

        Returns:
        --------
        Optional[Order]
            The order if successfully placed, None if rejected
        """
        # Validate order parameters
        if quantity <= 0:
            logger.warning("Order rejected: Invalid quantity")
            return None

        # Validate order type requirements
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            logger.warning("Order rejected: Limit price required")
            return None

        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            logger.warning("Order rejected: Stop price required")
            return None

        if order_type == OrderType.TRAILING_STOP:
            if trail_amount is None and trail_percent is None:
                logger.warning("Order rejected: Trail amount or percent required")
                return None

        # Validate based on side
        if side == OrderSide.SHORT and not self.allow_shorts:
            logger.warning("Order rejected: Short selling not allowed")
            return None

        # For market orders, check immediately if we can execute
        if order_type == OrderType.MARKET and self.current_bar:
            current_price = self.current_bar.close

            # Check affordability
            if side in [OrderSide.BUY, OrderSide.SHORT]:
                if not self._can_afford(quantity, current_price, side):
                    logger.warning("Order rejected: Insufficient funds")
                    return None

            # Check position availability
            if side == OrderSide.SELL and not self._has_position_to_sell(quantity):
                logger.warning("Order rejected: Insufficient position to sell")
                return None

            if side == OrderSide.COVER and not self._has_short_to_cover(quantity):
                logger.warning("Order rejected: No short position to cover")
                return None

        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            time_created=(
                self.current_bar.timestamp if self.current_bar else datetime.now()
            ),
            status=OrderStatus.PENDING,
        )

        self.orders[order.order_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders"""
        cancelled = 0
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                cancelled += 1
        return cancelled

    def liquidate(self) -> Optional[Order]:
        """
        Liquidate all positions at market price

        Returns:
        --------
        Optional[Order]
            The liquidation order if position exists, None otherwise
        """
        # Cancel all pending orders first
        self.cancel_all_orders()

        # Check if we have a position to liquidate
        if not self.position:
            return None

        # Determine order side based on position
        if self.position.quantity > 0:
            # Long position - sell to close
            return self.place_order(
                side=OrderSide.SELL,
                quantity=self.position.quantity,
                order_type=OrderType.MARKET,
            )
        else:
            # Short position - buy to cover
            return self.place_order(
                side=OrderSide.COVER,
                quantity=abs(self.position.quantity),
                order_type=OrderType.MARKET,
            )

    def _execute_trade(self, order: Order, execution_price: float) -> Trade:
        """Execute a trade and update positions"""
        # Calculate commission and slippage
        commission = self._calculate_commission(order.quantity, execution_price)
        slippage = self._calculate_slippage(execution_price, order.side)
        final_price = execution_price + slippage

        # Create trade record
        trade = Trade(
            trade_id=self._generate_trade_id(),
            order_id=order.order_id,
            symbol=self.symbol,
            side=order.side,
            quantity=order.quantity,
            price=final_price,
            commission=commission,
            slippage=slippage,
            timestamp=(
                self.current_bar.timestamp if self.current_bar else datetime.now()
            ),
        )

        # Update cash and position
        trade_value = order.quantity * final_price

        if order.side == OrderSide.BUY:
            # Deduct cash
            self.cash -= trade_value + commission

            # Update or create position
            if self.position and self.position.quantity > 0:
                # Add to existing long position
                total_quantity = self.position.quantity + order.quantity
                total_value = (
                    self.position.quantity * self.position.average_price
                ) + trade_value
                self.position.quantity = total_quantity
                self.position.average_price = total_value / total_quantity
                self.position.commission_paid += commission
            elif self.position and self.position.quantity < 0:
                # Covering a short position
                if order.quantity >= abs(self.position.quantity):
                    # Full cover and possibly go long
                    cover_quantity = abs(self.position.quantity)
                    remaining = order.quantity - cover_quantity

                    # Calculate realized P&L on the covered portion
                    cover_pnl = cover_quantity * (
                        self.position.average_price - final_price
                    )
                    self.position.realized_pnl += cover_pnl

                    if remaining > 0:
                        # Go long with remaining
                        self.position.quantity = remaining
                        self.position.average_price = final_price
                    else:
                        # Position closed
                        self.position = None
                else:
                    # Partial cover
                    self.position.quantity += order.quantity
                    cover_pnl = order.quantity * (
                        self.position.average_price - final_price
                    )
                    self.position.realized_pnl += cover_pnl
                    self.position.commission_paid += commission
            else:
                # Create new long position
                self.position = Position(
                    symbol=self.symbol,
                    quantity=order.quantity,
                    average_price=final_price,
                    current_price=final_price,
                    commission_paid=commission,
                )

        elif order.side == OrderSide.SELL:
            # Add cash
            self.cash += trade_value - commission

            if self.position and self.position.quantity > 0:
                if order.quantity >= self.position.quantity:
                    # Close entire position
                    realized_pnl = self.position.quantity * (
                        final_price - self.position.average_price
                    )
                    self.position.realized_pnl += realized_pnl
                    self.position = None
                else:
                    # Partial sale
                    realized_pnl = order.quantity * (
                        final_price - self.position.average_price
                    )
                    self.position.realized_pnl += realized_pnl
                    self.position.quantity -= order.quantity
                    self.position.commission_paid += commission

        elif order.side == OrderSide.SHORT:
            # Short selling
            if self.allow_shorts:
                # Add cash from short sale
                self.cash += trade_value - commission

                # Deduct margin requirement
                margin_required = trade_value * self.margin_requirement
                self.cash -= margin_required

                if self.position and self.position.quantity < 0:
                    # Add to existing short
                    total_quantity = self.position.quantity - order.quantity
                    total_value = (
                        abs(self.position.quantity) * self.position.average_price
                    ) + trade_value
                    self.position.quantity = total_quantity
                    self.position.average_price = total_value / abs(total_quantity)
                    self.position.commission_paid += commission
                else:
                    # Create new short position
                    self.position = Position(
                        symbol=self.symbol,
                        quantity=-order.quantity,
                        average_price=final_price,
                        current_price=final_price,
                        commission_paid=commission,
                    )

        elif order.side == OrderSide.COVER:
            # Covering a short
            if self.position and self.position.quantity < 0:
                # Deduct cash to buy back
                self.cash -= trade_value + commission

                # Return margin
                margin_return = (
                    order.quantity
                    * self.position.average_price
                    * self.margin_requirement
                )
                self.cash += margin_return

                if order.quantity >= abs(self.position.quantity):
                    # Full cover
                    cover_pnl = abs(self.position.quantity) * (
                        self.position.average_price - final_price
                    )
                    self.position.realized_pnl += cover_pnl
                    self.position = None
                else:
                    # Partial cover
                    cover_pnl = order.quantity * (
                        self.position.average_price - final_price
                    )
                    self.position.realized_pnl += cover_pnl
                    self.position.quantity += order.quantity
                    self.position.commission_paid += commission

        # Update order
        order.status = OrderStatus.FILLED
        order.fill_price = final_price
        order.filled_quantity = order.quantity
        order.time_filled = trade.timestamp
        order.commission = commission
        order.slippage = slippage

        # Record trade
        self.trades.append(trade)

        return trade

    def process_bar(self, bar: Bar) -> List[Trade]:
        """
        Process a new bar and execute any triggered orders

        Parameters:
        -----------
        bar : Bar
            New OHLCV bar

        Returns:
        --------
        List[Trade]
            List of executed trades
        """
        self.current_bar = bar
        executed_trades = []

        # Update position current price
        if self.position:
            self.position.current_price = bar.close

        # Process pending orders
        for order in list(self.orders.values()):
            if order.status != OrderStatus.PENDING:
                continue

            executed = False
            execution_price = None

            if order.order_type == OrderType.MARKET:
                # Execute at current bar close
                execution_price = bar.close
                executed = True

            elif order.order_type == OrderType.LIMIT:
                # Check if limit price was hit
                if order.price is not None:
                    if order.side in [OrderSide.BUY, OrderSide.COVER]:
                        if bar.low <= order.price:
                            execution_price = min(order.price, bar.close)
                            executed = True
                    else:  # SELL or SHORT
                        if bar.high >= order.price:
                            execution_price = max(order.price, bar.close)
                            executed = True

            elif order.order_type == OrderType.STOP:
                # Check if stop was triggered
                if order.stop_price is not None:
                    if order.side in [OrderSide.BUY, OrderSide.COVER]:
                        if bar.high >= order.stop_price:
                            execution_price = max(order.stop_price, bar.close)
                            executed = True
                    else:  # SELL or SHORT
                        if bar.low <= order.stop_price:
                            execution_price = min(order.stop_price, bar.close)
                            executed = True

            elif order.order_type == OrderType.STOP_LIMIT:
                # Check if stop was triggered
                if order.stop_price is not None and order.price is not None:
                    triggered = False
                    if order.side in [OrderSide.BUY, OrderSide.COVER]:
                        if bar.high >= order.stop_price:
                            triggered = True
                    else:  # SELL or SHORT
                        if bar.low <= order.stop_price:
                            triggered = True

                    # If triggered, check limit
                    if triggered:
                        if order.side in [OrderSide.BUY, OrderSide.COVER]:
                            if bar.low <= order.price:
                                execution_price = min(order.price, bar.close)
                                executed = True
                        else:  # SELL or SHORT
                            if bar.high >= order.price:
                                execution_price = max(order.price, bar.close)
                                executed = True

            elif order.order_type == OrderType.TRAILING_STOP:
                # Update trailing stop reference
                if order._trail_reference_price is None:
                    order._trail_reference_price = bar.close

                # Update reference price based on side
                if order.side in [OrderSide.SELL, OrderSide.SHORT]:
                    # For sell stops, track the highest price
                    order._trail_reference_price = max(
                        order._trail_reference_price, bar.high
                    )

                    # Calculate stop trigger
                    if order.trail_amount:
                        trigger_price = (
                            order._trail_reference_price - order.trail_amount
                        )
                    elif order.trail_percent:
                        trigger_price = order._trail_reference_price * (
                            1 - order.trail_percent
                        )
                    else:
                        continue

                    if bar.low <= trigger_price:
                        execution_price = min(trigger_price, bar.close)
                        executed = True

                else:  # BUY or COVER trailing stops
                    # For buy stops, track the lowest price
                    order._trail_reference_price = min(
                        order._trail_reference_price, bar.low
                    )

                    # Calculate stop trigger
                    if order.trail_amount:
                        trigger_price = (
                            order._trail_reference_price + order.trail_amount
                        )
                    elif order.trail_percent:
                        trigger_price = order._trail_reference_price * (
                            1 + order.trail_percent
                        )
                    else:
                        continue

                    if bar.high >= trigger_price:
                        execution_price = max(trigger_price, bar.close)
                        executed = True

            # Execute trade if triggered
            if executed and execution_price:
                # Final validation before execution
                can_execute = True

                if order.side in [OrderSide.BUY, OrderSide.SHORT]:
                    if not self._can_afford(
                        order.quantity, execution_price, order.side
                    ):
                        order.status = OrderStatus.REJECTED
                        can_execute = False
                elif order.side == OrderSide.SELL:
                    if not self._has_position_to_sell(order.quantity):
                        order.status = OrderStatus.REJECTED
                        can_execute = False
                elif order.side == OrderSide.COVER:
                    if not self._has_short_to_cover(order.quantity):
                        order.status = OrderStatus.REJECTED
                        can_execute = False

                if can_execute:
                    trade = self._execute_trade(order, execution_price)
                    executed_trades.append(trade)

        # Record portfolio state
        self._record_portfolio_state()

        return executed_trades

    def _record_portfolio_state(self):
        """Record current portfolio state"""
        positions = {}
        if self.position:
            positions[self.symbol] = self.position

        equity = self.get_equity()
        margin_used = 0.0

        if self.position and self.position.quantity < 0:
            # Calculate margin used for shorts
            margin_used = abs(
                self.position.quantity
                * self.position.average_price
                * self.margin_requirement
            )

        buying_power = self.cash - margin_used

        state = PortfolioState(
            timestamp=(
                self.current_bar.timestamp if self.current_bar else datetime.now()
            ),
            cash=self.cash,
            positions=positions.copy(),
            equity=equity,
            margin_used=margin_used,
            buying_power=buying_power,
        )

        self.portfolio_history.append(state)

    def get_position(self) -> Optional[Position]:
        """Get current position"""
        return self.position

    def get_cash(self) -> float:
        """Get current cash balance"""
        return self.cash

    def get_equity(self) -> float:
        """Get total portfolio equity (cash + positions)"""
        equity = self.cash

        if self.position:
            equity += self.position.market_value

            # Add back margin for shorts
            if self.position.quantity < 0:
                margin_used = abs(
                    self.position.quantity
                    * self.position.average_price
                    * self.margin_requirement
                )
                equity += margin_used

        return equity

    def get_buying_power(self) -> float:
        """Get available buying power"""
        if self.position and self.position.quantity < 0:
            margin_used = abs(
                self.position.quantity
                * self.position.average_price
                * self.margin_requirement
            )
            return self.cash - margin_used
        return self.cash

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders, optionally filtered by status"""
        if status:
            return [o for o in self.orders.values() if o.status == status]
        return list(self.orders.values())

    def get_trades(self) -> List[Trade]:
        """Get all executed trades"""
        return self.trades.copy()

    def get_portfolio_history(self) -> List[PortfolioState]:
        """Get portfolio state history"""
        return self.portfolio_history.copy()

    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_balance
        self.position = None
        self.orders.clear()
        self.trades.clear()
        self.order_counter = 0
        self.trade_counter = 0
        self.current_bar = None
        self.portfolio_history.clear()

    def get_performance_metrics(self, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate performance metrics

        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate for Sharpe/Sortino calculations

        Returns:
        --------
        Dict[str, float]
            Dictionary containing performance metrics
        """
        if len(self.portfolio_history) < 2:
            return {
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "total_commission": 0.0,
            }

        # Calculate returns series
        equity_series = [state.equity for state in self.portfolio_history]
        returns = []
        for i in range(1, len(equity_series)):
            ret = (equity_series[i] - equity_series[i - 1]) / equity_series[i - 1]
            returns.append(ret)

        # Total return
        total_return = equity_series[-1] - self.initial_balance
        total_return_pct = total_return / self.initial_balance * 100

        # Trade statistics - count round trips
        trade_returns = []
        winning_trades = 0
        losing_trades = 0
        total_commission = 0.0

        # Track buy trades to match with sells
        buy_trades = []

        for trade in self.trades:
            total_commission += trade.commission

            if trade.side == OrderSide.BUY:
                buy_trades.append(trade)
            elif trade.side == OrderSide.SELL and buy_trades:
                # Match with oldest buy (FIFO)
                buy_trade = buy_trades.pop(0)
                trade_return = (
                    (trade.price - buy_trade.price) * trade.quantity
                    - trade.commission
                    - buy_trade.commission
                )
                trade_returns.append(trade_return)
                if trade_return > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

        # Win rate
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (999.99 if gross_profit > 0 else 0.0)
        )

        # Average win/loss
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Drawdown calculation
        peak = equity_series[0]
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

        for equity in equity_series:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = drawdown / peak if peak > 0 else 0.0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct * 100

        # Risk metrics (simplified daily calculations)
        if len(returns) > 1:
            # Assume daily returns
            trading_days = 252

            avg_return = sum(returns) / len(returns)
            returns_std = math.sqrt(
                sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
            )

            # Sharpe ratio
            daily_rf = risk_free_rate / trading_days
            excess_returns = [r - daily_rf for r in returns]
            avg_excess_return = sum(excess_returns) / len(excess_returns)
            sharpe_ratio = (
                math.sqrt(trading_days) * avg_excess_return / returns_std
                if returns_std > 0
                else 0.0
            )

            # Sortino ratio (downside deviation)
            downside_returns = [min(0, r - daily_rf) for r in returns]
            downside_std = math.sqrt(
                sum(r**2 for r in downside_returns) / len(downside_returns)
            )
            sortino_ratio = (
                math.sqrt(trading_days) * avg_excess_return / downside_std
                if downside_std > 0
                else 0.0
            )

            # Calmar ratio
            annual_return = total_return_pct / 100  # Convert back to decimal
            calmar_ratio = (
                annual_return / (max_drawdown_pct / 100)
                if max_drawdown_pct > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0

        return {
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "calmar_ratio": round(calmar_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "total_commission": round(total_commission, 2),
        }


class AlpacaStockPortfolioEngine(PortfolioEngine):
    """
    Portfolio engine configured for Alpaca Markets - US Stocks

    Alpaca stock commission structure:
    - No commission for US stocks
    - No base fee
    - Regulatory fees only (very small)

    Realistic slippage for liquid stocks
    """

    def __init__(self, initial_balance: float = 100000.0, symbol: str = "AAPL"):
        super().__init__(
            initial_balance=initial_balance,
            commission_rate=0.0,  # No commission
            commission_pct=0.0,  # No percentage fee
            slippage_pct=0.0002,  # 0.02% slippage (2 basis points)
            slippage_fixed=0.01,  # 1 cent fixed slippage
            allow_shorts=False,  # Shorting disabled by default
            margin_requirement=0.5,  # 50% margin requirement (Reg T)
            symbol=symbol,
        )


class AlpacaCryptoPortfolioEngine(PortfolioEngine):
    """
    Portfolio engine configured for Alpaca Markets - Cryptocurrency

    Alpaca crypto commission structure:
    - 0.25% commission on all trades
    - No base fee

    Higher slippage for crypto markets
    """

    def __init__(self, initial_balance: float = 100000.0, symbol: str = "BTC/USD"):
        super().__init__(
            initial_balance=initial_balance,
            commission_rate=0.0,  # No flat fee
            commission_pct=0.0025,  # 0.25% commission
            slippage_pct=0.0005,  # 0.05% slippage (5 basis points)
            slippage_fixed=0.0,  # No fixed slippage for crypto
            allow_shorts=False,  # No shorting for crypto on Alpaca
            margin_requirement=0.0,  # N/A for spot trading
            symbol=symbol,
        )


# Keep the old class name for backward compatibility
AlpacaPortfolioEngine = AlpacaStockPortfolioEngine


class CoinbasePortfolioEngine(PortfolioEngine):
    """
    Portfolio engine configured for Coinbase

    Coinbase fee structure (for retail):
    - Maker fee: 0.40% (limit orders that add liquidity)
    - Taker fee: 0.60% (market orders or limit orders that remove liquidity)
    - Using taker fee as default since we're mostly using market orders

    Higher slippage due to crypto volatility
    """

    def __init__(self, initial_balance: float = 100000.0, symbol: str = "BTC-USD"):
        super().__init__(
            initial_balance=initial_balance,
            commission_rate=0.0,  # No flat fee
            commission_pct=0.006,  # 0.60% taker fee
            slippage_pct=0.001,  # 0.10% slippage (10 basis points)
            slippage_fixed=0.0,  # No fixed slippage for crypto
            allow_shorts=False,  # Coinbase spot doesn't allow shorting
            margin_requirement=0.0,  # N/A for spot trading
            symbol=symbol,
        )

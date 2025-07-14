"""
Alpaca Trading Client
Clean and modernized version with proper error handling and type safety.
"""

import time
import statistics
from typing import Optional, TypeVar, Callable
from uuid import UUID
from datetime import datetime, timedelta, timezone
from textwrap import indent

# Type variable for generic return types
T = TypeVar('T')

# Third-party imports
from statsmodels.tsa.stattools import adfuller

from alpaca.common.exceptions import APIError
from alpaca.data.historical.crypto import (
    CryptoHistoricalDataClient,
    CryptoLatestQuoteRequest,
    CryptoSnapshotRequest,
)
from alpaca.data.models.bars import Bar, BarSet
from alpaca.data.models.quotes import Quote
from alpaca.data.models.snapshots import Snapshot
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import CryptoBarsRequest
from alpaca.trading.client import TradingClient, TradeAccount, Asset
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    QueryOrderStatus,
)
from alpaca.trading.models import Order, Position
from alpaca.trading.requests import (
    LimitOrderRequest,
    StopLimitOrderRequest,
    GetOrdersRequest,
    ReplaceOrderRequest,
    MarketOrderRequest,
)

import requests

from chalk import red


class AccountInfo:
    """Wrapper for account information with clean interface"""

    def __init__(self, acct) -> None:
        self.equity_usd: float = float(acct.equity)
        self.available_usd: float = float(acct.cash)
        self.pattern_day_trader: bool = acct.pattern_day_trader or False
        self.daytrade_count: int = int(acct.daytrade_count or 0)
        self.trading_blocked: bool = (
            acct.trading_blocked or acct.trade_suspended_by_user
        )


class PositionInfo:
    """Wrapper for position information with clean interface"""

    def __init__(self, pos) -> None:
        self.total_qty: float = float(pos.qty)

        # Calculate entry price using worst-case scenario
        # https://docs.alpaca.markets/docs/position-average-entry-price-calculation
        avg_entry_price: float = float(pos.avg_entry_price)
        compressed_fifo: float = float(pos.cost_basis) / self.total_qty

        # Use MAX of both methods for safer profit targets
        # (Alpaca uses both weighted-average & compressed-fifo, depending on whether intraday or eod
        self.entry_price: float = max(avg_entry_price, compressed_fifo)
        self.qty_available: float = float(pos.qty_available) if pos.qty_available else 0.0


class PriceInfo:
    """Comprehensive price information with statistical analysis"""

    def __init__(self, minute, quote) -> None:
        # BUY Limit and Buy Stop orders are triggered when the ASK reaches your order's price.
        # SELL Limit and Sell Stop orders are triggered when the BID reaches your order's price.
        self.time: datetime = minute.timestamp

        # OHLC data
        self.high: float = minute.high
        self.close: float = minute.close
        self.low: float = minute.low
        self.hlc3: float = (minute.high + minute.low + minute.close) / 3.0

        # Bid/Ask data
        self.buy_ask: float = quote.ask_price
        self.sell_bid: float = quote.bid_price
        self.bid_ask_midpoint: float = (self.sell_bid + self.buy_ask) / 2.0
        self.bid_ask_range: float = abs(self.buy_ask - self.sell_bid)

        # Statistical analysis
        samples = [self.high, self.low, self.close, self.sell_bid, self.buy_ask]
        self.norm_med = statistics.median(samples)  # Median less affected by outliers
        self.norm_mod = statistics.mode(samples)
        self.norm_std = statistics.stdev(samples)
        self.norm_high_2 = self.norm_med + self.norm_std * 2
        self.norm_high_1 = self.norm_med + self.norm_std
        self.norm_low_1 = self.norm_med - self.norm_std
        self.norm_low_2 = self.norm_med - self.norm_std * 2


class OrderInfo:
    """Wrapper for order information with clean interface"""
    
    def __init__(self, order) -> None:
        if order is None:
            raise ValueError("Order cannot be None")
        
        self.id: UUID = order.id
        self.client_order_id: str = order.client_order_id
        self.symbol: str = order.symbol
        self.qty = float(order.qty) if order.qty else 0.0
        self.type = order.order_type
        self.side = order.side
        self.time_in_force = order.time_in_force

        self.limit_price = float(order.limit_price) if order.limit_price else 0.0
        stop: float = float(order.stop_price) if order.stop_price else 0.0
        self.stop_price = stop if stop > 0 else self.limit_price

        self.filled_qty: float = float(order.filled_qty) if order.filled_qty else 0.0
        self.filled_avg_price: float = (
            float(order.filled_avg_price) if order.filled_avg_price else 0.0
        )
        self.remaining_qty: float = self.qty - self.filled_qty
        self.status = order.status


class AlpacaClient:
    """
    Modern Alpaca trading client with comprehensive error handling and retry logic.
    Supports both crypto and equity trading with advanced order management.
    """

    def __init__(
        self,
        data_client,
        trade_client,
        symbol: str
    ):
        self.data_client = data_client
        self.trade_client = trade_client
        self.symbol: str = symbol

    def set_symbol(self, symbol: str) -> None:
        """Update the trading symbol"""
        self.symbol = symbol

    def get_account(self) -> AccountInfo:
        """Get account information with retry logic"""
        return self._retry_api_call(
            lambda: AccountInfo(self.trade_client.get_account()),
            "get account"
        )

    def get_snapshot(self, symbol: str | None = None) -> Snapshot:
        """Get latest market snapshot"""
        symbol = symbol or self.symbol
        req = CryptoSnapshotRequest(symbol_or_symbols=symbol)
        return self._retry_api_call(
            lambda: self.data_client.get_crypto_snapshot(req)[symbol],
            "get snapshot"
        )

    def get_latest_quote(self, symbol: str | None = None) -> Quote:
        """Get latest quote data"""
        symbol = symbol or self.symbol
        req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
        return self._retry_api_call(
            lambda: self.data_client.get_crypto_latest_quote(req)[symbol],
            "get latest quote"
        )

    def get_pricing(self, symbol: str | None = None) -> PriceInfo:
        """Get comprehensive pricing information"""
        snap = self.get_snapshot(symbol=symbol)
        quote = snap.latest_quote
        minute = snap.minute_bar
        return PriceInfo(minute=minute, quote=quote)

    def get_position(self, symbol: str | None = None) -> Optional[PositionInfo]:
        """Get position information for symbol"""
        symbol = symbol or self.symbol

        def _get_position():
            positions = self.trade_client.get_all_positions()
            matching_positions = [
                p for p in positions if p.symbol == symbol.replace("/", "")
            ]

            if not matching_positions:
                return None

            pos = PositionInfo(matching_positions[0])
            if pos.total_qty > 0:
                min_increment = self.get_min_trade_increment()
                if pos.total_qty >= min_increment:
                    return pos
            return None

        return self._retry_api_call(_get_position, "get position")

    def get_min_order_qty(self, symbol: str | None = None) -> float:
        """Get minimum order quantity for symbol"""
        symbol = symbol or self.symbol
        return self._retry_api_call(
            lambda: self.trade_client.get_asset(symbol.replace("/", "")).min_order_size,
            "get min order qty"
        )

    def get_min_trade_increment(self, symbol: str | None = None) -> float:
        """Get minimum trade increment for symbol"""
        symbol = symbol or self.symbol
        return self._retry_api_call(
            lambda: self.trade_client.get_asset(symbol.replace("/", "")).min_trade_increment,
            "get min trade increment"
        )

    def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get order by ID"""
        def _get_order():
            try:
                order = self.trade_client.get_order_by_id(order_id=order_id)
                return OrderInfo(order) if order else None
            except APIError as e:
                if e.code == 40410000:  # Order not found
                    return None
                raise

        return self._retry_api_call(_get_order, "get order")

    def get_open_limit_orders(self, side=None, symbol: str | None = None) -> list[OrderInfo]:
        """Get open limit orders"""
        symbol = symbol or self.symbol
        req = GetOrdersRequest(
            symbols=[symbol], status=QueryOrderStatus.OPEN, side=side
        )

        def _get_orders():
            orders = self.trade_client.get_orders(req)
            return [
                OrderInfo(o) for o in orders
                if o.type in [OrderType.STOP_LIMIT, OrderType.LIMIT]
            ]

        return self._retry_api_call(_get_orders, "get open orders") or []

    def submit_market_order(
        self,
        side: OrderSide,
        order_qty: float | None = None,
        order_usd: float | None = None,
        wait_seconds: float = 0.5,
        symbol: str | None = None,
    ) -> OrderInfo:
        """Submit market order"""
        symbol = symbol or self.symbol
        notional = round(order_usd, 2) if order_usd else None

        with open("trading.audit.log", "a") as f:
            f.write(f"{datetime.now()} {side.upper()} {symbol}\\n")

            req = MarketOrderRequest(
                symbol=symbol,
                side=side,
                type=OrderType.MARKET,
                notional=notional,
                qty=order_qty,
                time_in_force=TimeInForce.GTC,
            )

            f.write(indent(
                f"{side.upper()} {req.type.upper()} with "
                f"{'$' + str(notional) if notional else 'qty ' + str(order_qty)}\\n",
                "  "
            ))

            try:
                order = self.trade_client.submit_order(req)
                f.write(indent(
                    f"{order.status.upper()} with total qty {order.qty}, "
                    f"filled {order.filled_qty} at $ {order.filled_avg_price}\\n",
                    "  "
                ))

                if wait_seconds > 0:
                    time.sleep(wait_seconds)

                updated_order = self.get_order(order.id)
                if updated_order:
                    order = updated_order
                    f.write(indent(
                        f"{order.status.upper()} with total qty {order.qty}, "
                        f"filled {order.filled_qty} at $ {order.filled_avg_price}\\n",
                        "  "
                    ))

                # Set limit price for market orders post-facto
                if order.filled_avg_price > 0:
                    order.limit_price = order.filled_avg_price

                f.write(indent("OK\\n\\n", "  "))
                return order

            except APIError as e:
                print(f"Market order error: {e}")
                f.write(indent(f"Market order error: {e}\\n\\n", "  "))
                raise

    def submit_limit_order(
        self,
        side: OrderSide,
        order_qty: float,
        limit_price: float,
        symbol: str | None = None,
    ) -> OrderInfo:
        """Submit limit order"""
        symbol = symbol or self.symbol

        with open("trading.audit.log", "a") as f:
            f.write(f"{datetime.now()} {side.upper()} {symbol}\\n")

            req = LimitOrderRequest(
                symbol=symbol,
                side=side,
                type=OrderType.LIMIT,
                qty=order_qty,
                limit_price=limit_price,
                time_in_force=TimeInForce.GTC,
            )

            f.write(indent(
                f"{side.upper()} {req.type.upper()} with qty {order_qty} "
                f"@ limit $ {limit_price}\\n",
                "  "
            ))

            try:
                order = self.trade_client.submit_order(req)
                f.write(indent(
                    f"{order.status.upper()} with total qty {order.qty}, "
                    f"filled {order.filled_qty} at $ {order.filled_avg_price}\\n",
                    "  "
                ))

                f.write(indent("OK\\n\\n", "  "))
                return OrderInfo(order)

            except APIError as e:
                print(f"Limit order error: {e}")
                f.write(indent(f"Limit order error: {e}\\n\\n", "  "))
                raise

    def submit_stop_limit_order(
        self,
        side: OrderSide,
        order_qty: float,
        limit_price: float,
        stop_price: float = 0.0,
        symbol: str | None = None,
    ) -> OrderInfo:
        """Submit stop limit order"""
        symbol = symbol or self.symbol
        stop_price = stop_price if stop_price > 0 else limit_price

        with open("trading.audit.log", "a") as f:
            f.write(f"{datetime.now()} {side.upper()} {symbol}\\n")

            req = StopLimitOrderRequest(
                symbol=symbol,
                side=side,
                type=OrderType.STOP_LIMIT,
                qty=order_qty,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC,
            )

            f.write(indent(
                f"{side.upper()} {req.type.upper()} with qty {order_qty} "
                f"@ limit $ {limit_price} & stop $ {stop_price}\\n",
                "  "
            ))

            try:
                order = self.trade_client.submit_order(req)
                f.write(indent(
                    f"{order.status.upper()} with total qty {order.qty}, "
                    f"filled {order.filled_qty} at $ {order.filled_avg_price}\\n",
                    "  "
                ))

                f.write(indent("OK\\n\\n", "  "))
                return OrderInfo(order)

            except APIError as e:
                print(f"Stop Limit order error: {e}")
                f.write(indent(f"Stop Limit error: {e}\\n\\n", "  "))
                raise

    def cancel_order(self, order: Order | OrderInfo | str) -> bool:
        """Cancel an order"""
        with open("trading.audit.log", "a") as f:
            if isinstance(order, str):
                order_id = order
                f.write(f"{datetime.now()} CANCEL <order_id>\\n")
            else:
                order_id = order.id
                f.write(f"{datetime.now()} CANCEL {order.side.upper() if order.side else 'UNKNOWN'} {order.symbol}\\n")

            try:
                self.trade_client.cancel_order_by_id(order_id)
                f.write(indent("OK\\n\\n", "  "))
                return True
            except APIError as e:
                if e.code == 40410000:  # Order not found
                    f.write(indent("Order not found\\n\\n", "  "))
                    return True  # Idempotent
                elif e.code == 42210000:  # Already filled
                    f.write(indent("Order already filled\\n\\n", "  "))
                    return False
                else:
                    f.write(indent(f"Cancel error: {e}\\n\\n", "  "))
                    raise

    # Statistical analysis methods
    def get_stationary_analysis(self, bars: list[Bar], significance: float = 0.05) -> bool:
        """Perform stationarity analysis on price data"""
        if len(bars) < 4:
            return False

        prices = [bar.close for bar in bars]
        adf_result = adfuller(prices, autolag="AIC")
        p_value = adf_result[1]

        # p-value <= significance means stationary
        return bool(p_value <= significance)

    def get_daily_stationary(self, length: int = 5, significance: float = 0.05, symbol: str | None = None) -> bool:
        """Check if daily prices are stationary"""
        bars = self.get_day_bars(length=max(length, 4), symbol=symbol)
        return self.get_stationary_analysis(bars, significance)

    def get_hourly_stationary(self, length: int = 5, significance: float = 0.05, symbol: str | None = None) -> bool:
        """Check if hourly prices are stationary"""
        bars = self.get_hour_bars(length=max(length, 4), symbol=symbol)
        return self.get_stationary_analysis(bars, significance)

    # Bar data methods (simplified versions)
    def get_day_bars(self, length: int = 5, symbol: str | None = None) -> list[Bar]:
        """Get daily bars"""
        symbol = symbol or self.symbol
        length = max(length, 1)

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=length + 1)
        end = now + timedelta(minutes=1)

        day_timeframe: TimeFrame = TimeFrame.Day  # type: ignore
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=day_timeframe,
        )

        rsp = self._retry_api_call(
            lambda: self.data_client.get_crypto_bars(req),
            "get day bars"
        )

        if not rsp:
            return []

        bars = rsp[symbol][-length:]

        return bars

    def get_hour_bars(self, hours: int = 1, length: int = 23, symbol: str | None = None) -> list[Bar]:
        """Get hourly bars"""
        symbol = symbol or self.symbol
        length = max(length, 1)

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours * (length + 1))
        end = now + timedelta(minutes=1)

        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=TimeFrame(hours, TimeFrameUnit.Hour),  # type: ignore
        )

        rsp = self._retry_api_call(
            lambda: self.data_client.get_crypto_bars(req),
            "get hour bars"
        )

        if not rsp:
            return []

        bars = rsp[symbol][-length:]

        return bars

    def _retry_api_call(self, func: Callable[[], T], operation_name: str, max_retries: int = 3) -> T:
        """Generic retry wrapper for API calls"""
        last_exception = None
        for retry in range(max_retries, 0, -1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if isinstance(e, requests.exceptions.ConnectionError):
                    print(red(f"  {operation_name} retry {retry}"), flush=True)
                    if retry <= 1:
                        raise
                    time.sleep(3.0 / retry)
                else:
                    raise
        # This should never be reached, but just in case
        raise last_exception or Exception(f"Failed to {operation_name} after {max_retries} retries")

    @staticmethod
    def get_stop_price(side: OrderSide, limit_price: float, stop_price: float = 0.0) -> float:
        """Calculate stop price based on side and limit price"""
        if stop_price > 0.0:
            return stop_price

        if stop_price == 0.0:
            return limit_price

        # Negative stop_price is treated as delta from limit
        stop_delta = abs(stop_price)

        if side == OrderSide.BUY:
            return limit_price - stop_delta
        else:
            return limit_price + stop_delta

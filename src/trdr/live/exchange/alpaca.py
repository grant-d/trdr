"""Alpaca exchange implementation."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

from alpaca.common.exceptions import APIError
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.models.bars import Bar as AlpacaBar
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestBarRequest,
    StockBarsRequest,
    StockLatestBarRequest,
)
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide as AlpacaOrderSide,
)
from alpaca.trading.enums import (
    OrderStatus as AlpacaOrderStatus,
)
from alpaca.trading.enums import (
    OrderType as AlpacaOrderType,
)
from alpaca.trading.enums import (
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)

from ...core import Symbol, Timeframe
from ...data import TimeframeAdapter
from ..config import AlpacaCredentials
from .base import (
    AuthenticationError,
    ExchangeError,
    ExchangeInterface,
    HydraConnectionError,
    InsufficientFundsError,
    OrderRejectedError,
    RateLimitError,
)
from .types import (
    HydraAccountInfo,
    HydraBar,
    HydraFill,
    HydraOrderRequest,
    HydraOrderResponse,
    HydraOrderSide,
    HydraOrderStatus,
    HydraOrderType,
    HydraPositionInfo,
)

logger = logging.getLogger(__name__)


def _hydra_side_to_alpaca(side: HydraOrderSide) -> AlpacaOrderSide:
    """Convert Hydra order side to Alpaca."""
    return AlpacaOrderSide.BUY if side == HydraOrderSide.BUY else AlpacaOrderSide.SELL


def _alpaca_side_to_hydra(side: AlpacaOrderSide) -> HydraOrderSide:
    """Convert Alpaca order side to Hydra."""
    return HydraOrderSide.BUY if side == AlpacaOrderSide.BUY else HydraOrderSide.SELL


def _alpaca_status_to_hydra(status: AlpacaOrderStatus) -> HydraOrderStatus:
    """Convert Alpaca order status to Hydra."""
    mapping = {
        AlpacaOrderStatus.NEW: HydraOrderStatus.NEW,
        AlpacaOrderStatus.ACCEPTED: HydraOrderStatus.ACCEPTED,
        AlpacaOrderStatus.PENDING_NEW: HydraOrderStatus.PENDING_NEW,
        AlpacaOrderStatus.PARTIALLY_FILLED: HydraOrderStatus.PARTIALLY_FILLED,
        AlpacaOrderStatus.FILLED: HydraOrderStatus.FILLED,
        AlpacaOrderStatus.CANCELED: HydraOrderStatus.CANCELED,
        AlpacaOrderStatus.PENDING_CANCEL: HydraOrderStatus.PENDING_CANCEL,
        AlpacaOrderStatus.REJECTED: HydraOrderStatus.REJECTED,
        AlpacaOrderStatus.EXPIRED: HydraOrderStatus.EXPIRED,
    }
    return mapping.get(status, HydraOrderStatus.NEW)


def _alpaca_order_type_to_hydra(order_type: AlpacaOrderType) -> HydraOrderType:
    """Convert Alpaca order type to Hydra."""
    mapping = {
        AlpacaOrderType.MARKET: HydraOrderType.MARKET,
        AlpacaOrderType.LIMIT: HydraOrderType.LIMIT,
        AlpacaOrderType.STOP: HydraOrderType.STOP,
        AlpacaOrderType.STOP_LIMIT: HydraOrderType.STOP_LIMIT,
    }
    return mapping.get(order_type, HydraOrderType.MARKET)


def _alpaca_order_to_hydra(order: AlpacaOrder) -> HydraOrderResponse:
    """Convert Alpaca order to Hydra order response."""
    filled_at = None
    if order.filled_at:
        filled_at = (
            order.filled_at
            if isinstance(order.filled_at, datetime)
            else datetime.fromisoformat(str(order.filled_at))
        )

    submitted_at = order.submitted_at or datetime.now(timezone.utc)
    if not isinstance(submitted_at, datetime):
        submitted_at = datetime.fromisoformat(str(submitted_at))

    return HydraOrderResponse(
        order_id=str(order.id),
        client_order_id=order.client_order_id or "",
        symbol=order.symbol,
        side=_alpaca_side_to_hydra(order.side),
        qty=float(order.qty) if order.qty else 0.0,
        filled_qty=float(order.filled_qty) if order.filled_qty else 0.0,
        filled_avg_price=(float(order.filled_avg_price) if order.filled_avg_price else None),
        status=_alpaca_status_to_hydra(order.status),
        order_type=_alpaca_order_type_to_hydra(order.order_type),
        submitted_at=submitted_at,
        filled_at=filled_at,
        limit_price=float(order.limit_price) if order.limit_price else None,
        stop_price=float(order.stop_price) if order.stop_price else None,
    )


def _alpaca_bar_to_hydra(bar: AlpacaBar) -> HydraBar:
    """Convert Alpaca bar to Hydra bar."""
    timestamp = bar.timestamp
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()
    return HydraBar(
        timestamp=str(timestamp),
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume),
    )


class AlpacaExchange(ExchangeInterface):
    """Alpaca exchange implementation.

    Supports both paper and live trading with crypto and stocks.
    """

    def __init__(
        self,
        credentials: AlpacaCredentials,
        paper: bool = True,
    ):
        """Initialize Alpaca exchange.

        Args:
            credentials: API credentials
            paper: Use paper trading (default True for safety)
        """
        self._credentials = credentials
        self._paper = paper
        self._trade_client: TradingClient | None = None
        self._crypto_data_client: CryptoHistoricalDataClient | None = None
        self._stock_data_client: StockHistoricalDataClient | None = None
        self._fill_callback: Callable[[HydraFill], None] | None = None
        self._connected = False

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self._paper

    async def connect(self) -> None:
        """Establish connection to Alpaca."""
        if self._connected:
            return

        try:
            # Create trading client
            self._trade_client = TradingClient(
                api_key=self._credentials.api_key,
                secret_key=self._credentials.api_secret,
                paper=self._paper,
            )

            # Create data clients
            self._crypto_data_client = CryptoHistoricalDataClient(
                api_key=self._credentials.api_key,
                secret_key=self._credentials.api_secret,
            )

            self._stock_data_client = StockHistoricalDataClient(
                api_key=self._credentials.api_key,
                secret_key=self._credentials.api_secret,
            )

            # Mark connected before verification (so get_account works)
            self._connected = True

            # Verify connection by getting account
            try:
                await self.get_account()
                logger.info(f"Connected to Alpaca ({'paper' if self._paper else 'live'})")
            except Exception:
                self._connected = False
                raise

        except APIError as e:
            if "forbidden" in str(e).lower() or "unauthorized" in str(e).lower():
                raise AuthenticationError(f"Alpaca authentication failed: {e}")
            raise HydraConnectionError(f"Failed to connect to Alpaca: {e}")
        except Exception as e:
            raise HydraConnectionError(f"Failed to connect to Alpaca: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._trade_client = None
        self._crypto_data_client = None
        self._stock_data_client = None
        self._connected = False
        logger.info("Disconnected from Alpaca")

    def _ensure_connected(self) -> None:
        """Ensure exchange is connected."""
        if not self._connected or self._trade_client is None:
            raise HydraConnectionError("Not connected to Alpaca")

    async def get_account(self) -> HydraAccountInfo:
        """Get account information."""
        self._ensure_connected()
        try:
            account = self._trade_client.get_account()
            return HydraAccountInfo(
                equity=float(account.equity),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                currency="USD",
            )
        except APIError as e:
            raise ExchangeError(f"Failed to get account: {e}")

    async def get_positions(self) -> dict[str, HydraPositionInfo]:
        """Get all open positions."""
        self._ensure_connected()
        try:
            positions = self._trade_client.get_all_positions()
            result = {}
            for pos in positions:
                hydra_pos = self._convert_position(pos)
                result[hydra_pos.symbol] = hydra_pos
            return result
        except APIError as e:
            raise ExchangeError(f"Failed to get positions: {e}")

    async def get_position(self, symbol: str) -> HydraPositionInfo | None:
        """Get position for specific symbol."""
        self._ensure_connected()
        try:
            # Parse symbol using shared Symbol class
            sym = Symbol.parse(symbol)
            alpaca_symbol = sym.raw if sym.is_crypto else sym.raw.replace("/", "")

            positions = self._trade_client.get_all_positions()
            for pos in positions:
                if pos.symbol == alpaca_symbol:
                    return self._convert_position(pos)
            return None
        except APIError as e:
            if "position does not exist" in str(e).lower():
                return None
            raise ExchangeError(f"Failed to get position: {e}")

    def _convert_position(self, pos: AlpacaPosition) -> HydraPositionInfo:
        """Convert Alpaca position to Hydra position."""
        qty = float(pos.qty)
        return HydraPositionInfo(
            symbol=pos.symbol,
            side="long" if qty > 0 else "short",
            qty=abs(qty),
            avg_entry_price=float(pos.avg_entry_price),
            market_value=float(pos.market_value),
            unrealized_pnl=float(pos.unrealized_pl),
            qty_available=float(pos.qty_available) if pos.qty_available else abs(qty),
        )

    async def submit_order(self, order: HydraOrderRequest) -> HydraOrderResponse:
        """Submit order to Alpaca."""
        self._ensure_connected()

        # Parse symbol using shared Symbol class
        sym = Symbol.parse(order.symbol)
        alpaca_symbol = sym.raw if sym.is_crypto else sym.raw.replace("/", "")

        # Map time in force
        tif_map = {
            "gtc": TimeInForce.GTC,
            "day": TimeInForce.DAY,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        time_in_force = tif_map.get(order.time_in_force.lower(), TimeInForce.GTC)

        try:
            # Build order request based on type
            if order.order_type == HydraOrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=order.qty,
                    side=_hydra_side_to_alpaca(order.side),
                    time_in_force=time_in_force,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == HydraOrderType.LIMIT:
                if order.limit_price is None:
                    raise ValueError("Limit order requires limit_price")
                request = LimitOrderRequest(
                    symbol=alpaca_symbol,
                    qty=order.qty,
                    side=_hydra_side_to_alpaca(order.side),
                    time_in_force=time_in_force,
                    limit_price=order.limit_price,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == HydraOrderType.STOP:
                if order.stop_price is None:
                    raise ValueError("Stop order requires stop_price")
                request = StopOrderRequest(
                    symbol=alpaca_symbol,
                    qty=order.qty,
                    side=_hydra_side_to_alpaca(order.side),
                    time_in_force=time_in_force,
                    stop_price=order.stop_price,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == HydraOrderType.STOP_LIMIT:
                if order.stop_price is None or order.limit_price is None:
                    raise ValueError("Stop-limit order requires stop_price and limit_price")
                request = StopLimitOrderRequest(
                    symbol=alpaca_symbol,
                    qty=order.qty,
                    side=_hydra_side_to_alpaca(order.side),
                    time_in_force=time_in_force,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                    client_order_id=order.client_order_id,
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")

            # Submit order
            alpaca_order = self._trade_client.submit_order(request)
            return _alpaca_order_to_hydra(alpaca_order)

        except APIError as e:
            error_str = str(e).lower()
            if "insufficient" in error_str:
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            if "rejected" in error_str:
                raise OrderRejectedError(f"Order rejected: {e}")
            if "rate limit" in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            raise ExchangeError(f"Failed to submit order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        self._ensure_connected()
        try:
            self._trade_client.cancel_order_by_id(order_id)
            return True
        except APIError as e:
            if "not found" in str(e).lower():
                return False
            if "already" in str(e).lower():  # Already filled/cancelled
                return False
            raise ExchangeError(f"Failed to cancel order: {e}")

    async def get_order(self, order_id: str) -> HydraOrderResponse | None:
        """Get order status."""
        self._ensure_connected()
        try:
            order = self._trade_client.get_order_by_id(order_id)
            return _alpaca_order_to_hydra(order)
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise ExchangeError(f"Failed to get order: {e}")

    async def get_open_orders(self, symbol: str | None = None) -> list[HydraOrderResponse]:
        """Get all open orders."""
        self._ensure_connected()
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            if symbol:
                request = GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[symbol.replace("/", "")],
                )
            orders = self._trade_client.get_orders(request)
            return [_alpaca_order_to_hydra(o) for o in orders]
        except APIError as e:
            raise ExchangeError(f"Failed to get open orders: {e}")

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[HydraBar]:
        """Get historical bars.

        Note: For live trading, prefer using AlpacaDataClient.get_bars() which
        includes caching. This method is kept for interface compatibility.
        """
        self._ensure_connected()

        sym = Symbol.parse(symbol)
        tf = Timeframe.parse(timeframe)
        adapter = TimeframeAdapter(tf, sym)
        alpaca_tf = adapter.to_alpaca()
        alpaca_symbol = sym.raw if sym.is_crypto else sym.raw.replace("/", "")

        # Calculate start date based on timeframe and limit
        # Add 20% buffer for weekends/gaps
        tf_minutes = tf.canonical.to_minutes()
        start = datetime.now(timezone.utc) - timedelta(minutes=tf_minutes * limit * 1.2)

        try:
            if sym.is_crypto:
                request = CryptoBarsRequest(
                    symbol_or_symbols=sym.raw,
                    timeframe=alpaca_tf,
                    start=start,
                    limit=limit,
                )
                bars = self._crypto_data_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=alpaca_tf,
                    start=start,
                    limit=limit,
                )
                bars = self._stock_data_client.get_stock_bars(request)

            # Extract bars for symbol - BarSet stores data in .data attribute
            bars_data = bars.data if hasattr(bars, "data") else bars
            symbol_key = sym.raw if sym.is_crypto else alpaca_symbol
            if symbol_key in bars_data:
                return [_alpaca_bar_to_hydra(b) for b in bars_data[symbol_key]]
            return []

        except APIError as e:
            raise ExchangeError(f"Failed to get bars: {e}")

    async def get_latest_bar(self, symbol: str) -> HydraBar | None:
        """Get latest bar for symbol."""
        self._ensure_connected()

        sym = Symbol.parse(symbol)
        alpaca_symbol = sym.raw if sym.is_crypto else sym.raw.replace("/", "")

        try:
            if sym.is_crypto:
                request = CryptoLatestBarRequest(symbol_or_symbols=sym.raw)
                result = self._crypto_data_client.get_crypto_latest_bar(request)
            else:
                request = StockLatestBarRequest(symbol_or_symbols=alpaca_symbol)
                result = self._stock_data_client.get_stock_latest_bar(request)

            result_data = result.data if hasattr(result, "data") else result
            key = sym.raw if sym.is_crypto else alpaca_symbol
            if key in result_data:
                return _alpaca_bar_to_hydra(result_data[key])
            return None

        except APIError as e:
            raise ExchangeError(f"Failed to get latest bar: {e}")

    async def get_latest_price(self, symbol: str) -> float | None:
        """Get latest price for symbol."""
        bar = await self.get_latest_bar(symbol)
        return bar.close if bar else None

    def subscribe_fills(self, callback: Callable[[HydraFill], None]) -> None:
        """Subscribe to fill events."""
        self._fill_callback = callback
        # Note: WebSocket streaming would be implemented here for real-time fills
        logger.info("Fill subscription registered (polling mode)")

    def unsubscribe_fills(self) -> None:
        """Unsubscribe from fill events."""
        self._fill_callback = None

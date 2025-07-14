import time
import statistics
from typing import Optional
from uuid import UUID
from datetime import datetime, timedelta, timezone
from textwrap import indent
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
    def __init__(self, acct: TradeAccount):
        self.equity_usd: float = float(acct.equity)
        self.available_usd: float = float(acct.cash)
        self.pattern_day_trader: bool = acct.pattern_day_trader or False
        self.daytrade_count: int = int(acct.daytrade_count or 0)
        self.trading_blocked: bool = (
            acct.trading_blocked or acct.trade_suspended_by_user
        )


class PositionInfo:
    def __init__(self, pos: Position):
        self.total_qty: float = float(pos.qty)

        # https://docs.alpaca.markets/docs/position-average-entry-price-calculation#weighted-average
        avg_entry_price: float = float(pos.avg_entry_price)
        # https://docs.alpaca.markets/docs/position-average-entry-price-calculation#compressed-fifo-first-in-first-out
        compressed_fifo: float = (
            float(pos.cost_basis) / self.total_qty
        )  # Cost basis is extended cost, NOT per-share
        # Alpaca uses both weighted-average & compressed-fifo, depending on whether intraday or eod
        # So safer (in terms of profit target) to use worst-case cost, which is MAX of both
        self.entry_price: float = max(avg_entry_price, compressed_fifo)

        # self.current_price = float(pos.current_price) if pos.current_price else 0.0
        self.qty_available: float = (
            float(pos.qty_available) if pos.qty_available else 0.0
        )


class PriceInfo:
    def __init__(self, minute: Bar, quote: Quote, heikin_ashi: bool):
        # https://money.stackexchange.com/a/145434
        # BUY Limit and Buy Stop orders are triggered when the ASK reaches your order's price.
        # SELL Limit and Sell Stop orders are triggered when the BID reaches your order's price.
        self.time: datetime = minute.timestamp
        self.heikin_ashi: bool = heikin_ashi

        self.high: float = minute.high
        self.close: float = minute.close
        self.low: float = minute.low
        self.hlc3: float = (minute.high + minute.low + minute.close) / 3.0

        # bid <= ask
        self.buy_ask: float = quote.ask_price
        self.sell_bid: float = quote.bid_price

        # Computed
        self.bid_ask_midpoint: float = (self.sell_bid + self.buy_ask) / 2.0
        self.bid_ask_range: float = abs(self.buy_ask - self.sell_bid)

        samples: list[float] = [
            self.high,
            self.low,
            self.close,
            self.sell_bid,
            self.buy_ask,
        ]
        # self.norm_avg = statistics.mean(samples)  # Like hlc3 but more representative
        self.norm_med = statistics.median(
            samples
        )  # Median is less affected by outliers
        self.norm_mod = statistics.mode(samples)  # Mode even less affected by outliers
        # 68/95/98% = 1/2/3 stdev
        # SAMPLE not POP since using only 5 of N ticks
        self.norm_std = statistics.stdev(samples)  # SAMPLE stdev (68%)
        # self.norm_stdev_2 = norm.stdev * 2  # Sample stdev (95%)
        # self.norm_stdev_3 = norm.stdev * 3  # Sample stdev (98%)
        self.norm_high_2 = self.norm_med + self.norm_std * 2  # 2 stdev (95%)
        self.norm_high_1 = self.norm_med + self.norm_std  # 1 stdev (68%)
        self.norm_low_1 = self.norm_med - self.norm_std  # 1 stdev (68%)
        self.norm_low_2 = self.norm_med - self.norm_std * 2  # 2 stdev (95%)


class OrderInfo:
    def __init__(self, order: Order | None):
        self.id: UUID = order.id
        self.client_order_id: str = order.client_order_id
        self.symbol: str = order.symbol
        self.qty = float(order.qty) if order.qty else 0.0
        self.type: OrderType = order.order_type
        self.side: OrderSide = order.side
        self.time_in_force: TimeInForce = order.time_in_force

        self.limit_price = float(order.limit_price) if order.limit_price else 0.0
        # Default stop_price to limit_price
        stop: float = float(order.stop_price) if order.stop_price else 0.0
        self.stop_price = stop if stop > 0 else self.limit_price

        self.filled_qty: float = float(order.filled_qty) if order.filled_qty else 0.0
        self.filled_avg_price: float = (
            float(order.filled_avg_price) if order.filled_avg_price else 0.0
        )
        self.remaining_qty: float = self.qty - self.filled_qty

        self.status: OrderStatus = order.status


class AlpacaClient:

    def __init__(
        self,
        data_client: CryptoHistoricalDataClient,
        trade_client: TradingClient,
        symbol: str,
        heikin_ashi: bool,
        # lrsi_weight: float
    ):
        self.data_client: CryptoHistoricalDataClient = data_client
        self.trade_client: TradingClient = trade_client
        self.symbol: str = symbol
        self.heikin_ashi: bool = heikin_ashi
        # self.lrsi_weight: float = lrsi_weight

    def set_symbol(self, symbol: str = None):
        self.symbol = symbol

    def get_account(self) -> AccountInfo:
        for retry in range(3, 0, -1):
            try:
                acct: TradeAccount = self.trade_client.get_account()
                account = AccountInfo(acct)
                return account

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def get_snapshot(self, symbol: str = None) -> Snapshot:
        symbol = symbol or self.symbol
        req = CryptoSnapshotRequest(symbol_or_symbols=symbol)
        for retry in range(3, 0, -1):
            try:
                snap: Snapshot = self.data_client.get_crypto_snapshot(req)
                return snap[symbol]

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def get_latest_quote(self, symbol: str = None) -> Quote:
        symbol = symbol or self.symbol
        req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
        for retry in range(3, 0, -1):
            try:
                quote: Quote = self.data_client.get_crypto_latest_quote(req)
                return quote[symbol]

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    # Doesn't always  get latest bar (1m) but rather a few minutes old
    # def get_latest_bar(self, symbol: str = None) -> Bar:
    #     symbol = symbol or self.symbol
    #     req = CryptoLatestBarRequest(symbol_or_symbols=symbol)
    #     bar: Bar = self.data_client.get_crypto_latest_bar(req)
    #     return bar[symbol]

    def get_pricing(self, symbol: str = None) -> PriceInfo:
        snap: Snapshot = self.get_snapshot(symbol=symbol)
        quote: Quote = snap.latest_quote

        # Pricing is ALWAYS standard minute bars (NOT Heiken Ashi)
        # CONFIRMED: Latest snapshot bar always EQUAL or NEWER than latest 1-minute bar
        minute: Bar = snap.minute_bar


        pricing = PriceInfo(minute=minute, quote=quote, heikin_ashi=self.heikin_ashi)
        return pricing

    def get_day_bars(self, length: int = 5, raw=False, symbol: str = None) -> list[Bar]:
        symbol = symbol or self.symbol
        length = int(max(length, 1))  # Ensure at least 1 bar
        ln: int = int(
            length + 1 if self.heikin_ashi and not raw else length
        )  # Add 1 for HA lag

        now = datetime.now(timezone.utc)
        start: datetime = now - timedelta(days=ln + 1)  # Add 1 else get 1 too few
        end: datetime = now + timedelta(minutes=1)

        # Does not always return the current bar
        # https://forum.alpaca.markets/t/get-most-recent-bars/12024/2
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            # limit=length,  # BEWARE, truncates off END not start
            timeframe=TimeFrame.Day,
        )
        for retry in range(3, 0, -1):
            try:
                rsp: BarSet = self.data_client.get_crypto_bars(req)
                bars: list[Bar] = rsp[symbol]

                # Prune local length
                bars = bars[-ln:]

                # Convert to Heiken Ashi
                if self.heikin_ashi and not raw:
                    bars = self.__to_heiken_ashi(bars)

                # Prune to input length
                bars = bars[-length:]
                return bars

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def get_hour_bars(
        self, hours: int = 1, length: int = 23, raw=False, symbol: str = None
    ) -> list[Bar]:
        symbol = symbol or self.symbol
        length = int(max(length, 1))  # Ensure at least 1 bar
        ln: int = int(
            length + 1 if self.heikin_ashi and not raw else length
        )  # Add 1 for HA lag

        now = datetime.now(timezone.utc)
        start: datetime = now - timedelta(
            hours=hours * (ln + 1)
        )  # Add 1 else get 1 too few
        end: datetime = now + timedelta(minutes=1)

        # Does not always return the current bar
        # https://forum.alpaca.markets/t/get-most-recent-bars/12024/2
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            # limit=length,  # BEWARE, truncates off END not start
            timeframe=TimeFrame(hours, TimeFrameUnit.Hour),
        )
        for retry in range(3, 0, -1):
            try:
                rsp: BarSet = self.data_client.get_crypto_bars(req)
                bars: list[Bar] = rsp[symbol]

                # Prune local length
                bars = bars[-ln:]

                # Convert to Heiken Ashi
                if self.heikin_ashi and not raw:
                    bars = self.__to_heiken_ashi(bars)

                # Prune to input length
                bars = bars[-length:]
                return bars

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def get_minute_bars(
        self, minutes: int, length: int, raw=False, symbol: str = None
    ) -> list[Bar]:
        length = int(max(length, 1))  # Ensure at least 1 bar
        ln: int = int(
            length + 1 if self.heikin_ashi and not raw else length
        )  # Add 1 for HA lag

        # Get raw minute bars
        bars: list[Bar] = self.__get_raw_minute_bars(
            minutes=minutes, length=ln, symbol=symbol
        )

        # Convert to Heiken Ashi
        if self.heikin_ashi and not raw:
            bars = self.__to_heiken_ashi(bars)

        # Prune to input length
        bars = bars[-length:]
        return bars

    def __get_raw_minute_bars(
        self, minutes: int, length: int, symbol: str = None
    ) -> list[Bar]:
        symbol = symbol or self.symbol
        length = int(max(length, 1))  # Ensure at least 1 bar

        now = datetime.now(timezone.utc)
        start: datetime = now - timedelta(
            minutes=max(minutes * (length + 1), 15)
        )  # Saw no actitivity for ~10 mins
        end: datetime = now + timedelta(minutes=1)

        # Does not always return the current bar
        # https://forum.alpaca.markets/t/get-most-recent-bars/12024/2
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            # limit=ln,  # BEWARE, truncates off END not start
            timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        )
        for retry in range(3, 0, -1):
            try:
                rsp: BarSet = self.data_client.get_crypto_bars(req)
                bars: list[Bar] = rsp[symbol]

                # CONFIRMED: Latest snapshot bar always EQUAL or NEWER than latest 1-minute bar
                if minutes == 1:
                    snap: Snapshot = self.get_snapshot()
                    if len(bars) == 0:
                        print("---------------------")
                        print(red("Augment empty"), snap.minute_bar.close)
                        bars.append(snap.minute_bar)
                    elif snap.minute_bar.timestamp == bars[-1].timestamp:
                        if snap.minute_bar.close != bars[-1].close:
                            print("---------------------")
                            print(
                                red("Same time, different close"),
                                snap.minute_bar.close,
                                bars[-1].close,
                            )
                            bars[-1] = snap.minute_bar
                    elif snap.minute_bar.timestamp > bars[-1].timestamp:
                        # New bar detected - append to existing bars
                        print("---------------------")
                        print(red("New bar"), snap.minute_bar.timestamp, snap.minute_bar.close, bars[-1].timestamp, bars[-1].close)
                        bars.append(snap.minute_bar)

                # Prune to input length
                bars = bars[-length:]
                return bars

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def __to_heiken_ashi(self, bars: list[Bar]) -> list[Bar]:
        # print(bld("Converting to Heiken Ashi"), len(bars))
        ha_bars: list[Bar] = []

        for i in range(1, len(bars)):
            bar = bars[i]
            bar_1 = bars[i - 1]

            ha_close = (bar.open + bar.high + bar.low + bar.close) / 4
            ha_open = (bar_1.open + bar_1.close) / 2

            ha_bars.append(
                Bar(
                    symbol=bar.symbol,
                    raw_data={
                        "t": bar.timestamp,
                        "o": ha_open,
                        "h": max(bar.high, ha_open, ha_close),
                        "l": min(bar.low, ha_open, ha_close),
                        "c": ha_close,
                        "v": bar.volume,
                        "n": bar.trade_count,
                        "vw": bar.vwap,
                    },
                )
            )

        # print("hh", len(bars), len(ha_bars), bars[-1].close, ha_bars[-1].close)
        return ha_bars

    def get_daily_stationary(
        self, length: int = 5, significance=0.05, symbol: str = None
    ) -> bool:
        length = int(
            max(length, 4)
        )  # API Error: sample size is too short to use selected regression component
        bars: list[Bar] = self.get_day_bars(length=length, raw=True, symbol=symbol)
        stationary: bool = self.__get_stationary(bars=bars, significance=significance)
        return stationary

    # Keep it tight so we react quickly to price changes
    def get_hourly_stationary(
        self, length: int = 5, significance=0.05, symbol: str = None
    ) -> bool:
        length = int(
            max(length, 4)
        )  # API Error: sample size is too short to use selected regression component
        bars: list[Bar] = self.get_hour_bars(length=length, raw=True, symbol=symbol)
        stationary: bool = self.__get_stationary(bars=bars, significance=significance)
        return stationary

    def get_1_minute_stationary(
        self, length: int = 1 * 60, significance=0.05, symbol: str = None
    ) -> bool:  # 1H
        length = int(
            max(length, 4)
        )  # API Error: sample size is too short to use selected regression component
        bars: list[Bar] = self.get_minute_bars(
            minutes=1, length=length, raw=True, symbol=symbol
        )
        stationary: bool = self.__get_stationary(bars=bars, significance=significance)
        return stationary

    def get_5_minute_stationary(
        self, length: int = 2 * 60 // 5, significance=0.05, symbol: str = None
    ) -> bool:  # 2H
        length = int(
            max(length, 4)
        )  # API Error: sample size is too short to use selected regression component
        bars: list[Bar] = self.get_minute_bars(
            minutes=5, length=length, raw=True, symbol=symbol
        )
        stationary: bool = self.__get_stationary(bars=bars, significance=significance)
        return stationary

    def get_15_minute_stationary(
        self, length: int = 3 * 60 // 15, significance=0.05, symbol: str = None
    ) -> bool:  # 3H
        length = int(
            max(length, 4)
        )  # API Error: sample size is too short to use selected regression component
        bars: list[Bar] = self.get_minute_bars(
            minutes=15, length=length, raw=True, symbol=symbol
        )
        stationary: bool = self.__get_stationary(bars=bars, significance=significance)
        return stationary

    def __get_stationary(self, bars: list[Bar], significance=0.05) -> bool:
        n: int = len(bars)  # May return less rows than requested
        x = [bar.close for bar in bars[:n]]
        adf = adfuller(x, autolag="AIC")
        # s_val: float = adf[0]  # Test-statistic
        p_val: float = adf[1]  # The p-value
        # lags: float = adf[2]  # Number of lags used
        # nobs: float = adf[3]  # Number of observations used
        # c_val: list[float] = [adf[4]["1%"], adf[4]["5%"], adf[4]["10%"]]  # Critical-values
        # print(f"ADF: {p_val=}, {s_val=}, {c_val=}")
        # https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/
        # The p-value is GREATER than significance level 0.05 and ADF statistic is higher than any critical values.
        # So, the time series is NON-stationary.
        stationary: bool = (
            p_val <= significance
        )  # or s_val < c_val[1]  # 5% significance level
        return stationary

    def get_position(self, symbol: str = None) -> Optional[PositionInfo]:
        # API Error: 1 validation error for list[Position]
        # Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
        # For further information visit https://errors.pydantic.dev/2.8/v/list_type
        symbol = symbol or self.symbol
        for retry in range(3, 0, -1):
            try:
                positions: list[Position] = self.trade_client.get_all_positions()
                positions = [
                    p for p in positions if p.symbol == symbol.replace("/", "")
                ]
                if len(positions) > 0:
                    pos = PositionInfo(positions[0])
                    if pos.total_qty > 0:
                        # Check if position quantity is significant
                    min_trade_increment: float = self.get_min_trade_increment()
                    if pos.total_qty >= min_trade_increment:
                        return pos
                return None

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def get_min_order_qty(self, symbol: str = None) -> float:
        symbol = symbol or self.symbol
        for retry in range(3, 0, -1):
            try:
                asset: Asset = self.trade_client.get_asset(symbol.replace("/", ""))
                return asset.min_order_size

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def get_min_trade_increment(self, symbol: str = None) -> float:
        symbol = symbol or self.symbol
        for retry in range(3, 0, -1):
            try:
                asset: Asset = self.trade_client.get_asset(symbol.replace("/", ""))
                return asset.min_trade_increment

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def get_order(self, order_id: UUID | str) -> OrderInfo | None:
        for retry in range(3, 0, -1):
            try:
                order = self.trade_client.get_order_by_id(order_id=order_id)
                return OrderInfo(order) if order else None

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

            except APIError as e:
                if e.code == 40410000:
                    print("Order not found")
                    return None  # Not found

                print(f"Get order error: {e}")
                raise e

    def get_open_limit_orders(
        self, side: OrderSide = None, symbol: str = None
    ) -> list[OrderInfo]:
        symbol = symbol or self.symbol
        req = GetOrdersRequest(
            symbols=[symbol], status=QueryOrderStatus.OPEN, side=side
        )
        for retry in range(3, 0, -1):
            try:
                orders: list[Order] = self.trade_client.get_orders(req)
                orders = [
                    OrderInfo(o)
                    for o in orders
                    if o.type == OrderType.STOP_LIMIT or o.type == OrderType.LIMIT
                ]
                return orders

            except requests.exceptions.ConnectionError as e:
                print(red(f"  retry {retry}"), flush=True)
                if retry <= 1:
                    raise e
                time.sleep(3.0 / retry)  # 1, 1.5, 3

    def cancel_order(self, order: OrderInfo | str) -> bool:
        with open("martingale.audit.log", "a") as f:
            order_id: str = None
            if isinstance(order, str):
                order_id = order
                f.write(f"{datetime.now()} CANCEL <not specified>\n")
                f.write(indent(f"CANCEL id {order_id}\n", "  "))
            else:
                order_id = order.id
                f.write(
                    f"{datetime.now()} CANCEL {order.side.upper()} {order.symbol}\n"
                )
                f.write(
                    indent(
                        f"CANCEL {order.side.upper()} {order.type.upper()} with qty {order.qty} @ limit $ {order.limit_price} & stop $ {order.stop_price or order.limit_price}\n",
                        "  ",
                    )
                )

            for retry in range(3, 0, -1):
                try:
                    # Cancel order
                    self.trade_client.cancel_order_by_id(order_id)

                    f.write(indent("OK\n\n", "  "))
                    return True

                except requests.exceptions.ConnectionError as e:
                    # OK to retry since cancellation (by id) is idempotent
                    print(red(f"  retry {retry}"), flush=True)
                    if retry <= 1:
                        raise e
                    time.sleep(3.0 / retry)  # 1, 1.5, 3

                except APIError as e:
                    # Cancel order error: {"code":40410000,"message":"order not found"}
                    if e.code == 40410000:
                        print("Order not found")
                        f.write(indent("Alpaca: order not found\n\n", "  "))
                        return True  # Idempotent

                    # alpaca.common.exceptions.APIError: {"code":42210000,"message":"order is already in \"filled\" state"}
                    if e.code == 42210000:
                        print('Order is already in "filled" state')
                        f.write(
                            indent(
                                'Alpaca: order is already in "filled" state\n\n', "  "
                            )
                        )
                        return False  # Failed

                    print(f"Cancel order error: {e}")
                    f.write(indent(f"Cancel order error: {e}\n\n", "  "))
                    raise e

    def submit_market_order(
        self,
        side: OrderSide,
        order_qty: float = None,
        order_usd: float = None,
        wait_seconds: float = 0.5,
        symbol: str = None,
    ) -> OrderInfo:
        symbol = symbol or self.symbol
        # API Error: {"code":42210000,"message":"notional value must be limited to 2 decimal places"}
        notional: float = round(order_usd, 2) if order_usd else None

        with open("martingale.audit.log", "a") as f:
            f.write(f"{datetime.now()} {side.upper()} {symbol}\n")

            # Construct MARKET order
            req = MarketOrderRequest(
                symbol=symbol,
                side=side,
                type=OrderType.MARKET,
                notional=notional,
                qty=order_qty,
                time_in_force=TimeInForce.GTC,
            )
            f.write(
                indent(
                    f"{req.side.upper()} {req.type.upper()} with {f'$ {notional}' if notional else f'qty {order_qty}'}\n",
                    "  ",
                )
            )

            # for retry in range(3, 0, -1):
            try:
                # Submit order
                order: Order = self.trade_client.submit_order(req)
                f.write(
                    indent(
                        f"{order.status.upper()} with total qty {order.qty}, filled {order.filled_qty} at $ {order.filled_avg_price}\n",
                        "  ",
                    )
                )

                if wait_seconds > 0:
                    time.sleep(wait_seconds)

                order = self.get_order(order.id)
                f.write(
                    indent(
                        f"{order.status.upper()} with total qty {order.qty}, filled {order.filled_qty} at $ {order.filled_avg_price}\n",
                        "  ",
                    )
                )

                # Serenade as a LIMIT order post-facto
                if order.filled_avg_price > 0:
                    order.limit_price = order.filled_avg_price

                f.write(indent("OK\n\n", "  "))
                return order

            # except requests.exceptions.ConnectionError as e:
            #     print(red(f'  retry {retry}'), flush=True)
            #     if retry <= 1:
            #         raise e
            #     time.sleep(3.0 / retry)  # 1, 1.5, 3

            except APIError as e:
                print(f"Market order error: {e}")
                f.write(indent(f"Market order error: {e}\n\n", "  "))
                raise e

    def submit_limit_order(
        self,
        side: OrderSide,
        order_qty: float,
        limit_price: float,
        symbol: str = None,
    ) -> OrderInfo:
        symbol = symbol or self.symbol
        with open("martingale.audit.log", "a") as f:
            f.write(f"{datetime.now()} {side.upper()} {symbol}\n")

            # Construct LIMIT order
            req = LimitOrderRequest(
                symbol=symbol,
                side=side,
                type=OrderType.LIMIT,
                qty=order_qty,
                limit_price=limit_price,
                time_in_force=TimeInForce.GTC,
            )
            f.write(
                indent(
                    f"{req.side.upper()} {req.type.upper()} with qty {order_qty} @ limit $ {limit_price}\n",
                    "  ",
                )
            )

            # for retry in range(3, 0, -1):
            try:
                # Submit order
                order: Order = self.trade_client.submit_order(req)
                f.write(
                    indent(
                        f"{order.status.upper()} with total qty {order.qty}, filled {order.filled_qty} at $ {order.filled_avg_price}\n",
                        "  ",
                    )
                )

                f.write(indent("OK\n\n", "  "))
                return OrderInfo(order)

            # except requests.exceptions.ConnectionError as e:
            #     print(red(f'  retry {retry}'), flush=True)
            #     if retry <= 1:
            #         raise e
            #     time.sleep(3.0 / retry)  # 1, 1.5, 3

            except APIError as e:
                print(f"Limit order error: {e}")
                f.write(indent(f"Limit order error: {e}\n\n", "  "))
                raise e

    @staticmethod

    def get_stop_price(
        side: OrderSide,
        limit_price: float,
        stop_price: float = 0.0,  # Treat negative value as a delta
    ) -> float:
        # If STOP > 0.0, use it
        if stop_price > 0.0:
            return stop_price

        # If STOP = 0.0, use LIMIT
        if stop_price == 0.0:
            return limit_price

        # If STOP < 0.0, consider it a DELTA from LIMIT
        stop_price_delta: float = abs(stop_price)

        # If BUY, set STOP slightly BELOW desired LIMIT (ie BUY for slightly MORE than trigger)
        if side == OrderSide.BUY:
            return limit_price - stop_price_delta

        # If SELL, set STOP slightly ABOVE desired LIMIT (ie SELL for slightly LESS than trigger)
        return limit_price + stop_price_delta

    # Emulate trailing orders with STOP-LIMIT
    def submit_stop_limit_order(
        self,
        side: OrderSide,
        order_qty: float,
        limit_price: float,
        stop_price: float = 0.0,
        symbol: str = None,
    ) -> OrderInfo:
        symbol = symbol or self.symbol
        with open("martingale.audit.log", "a") as f:
            f.write(f"{datetime.now()} {side.upper()} {symbol}\n")

            # Ensure STOP price
            stop_price = stop_price if stop_price > 0 else limit_price

            # Construct STOP-LIMIT order
            req = StopLimitOrderRequest(
                symbol=symbol,
                side=side,
                type=OrderType.STOP_LIMIT,
                qty=order_qty,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC,
            )
            f.write(
                indent(
                    f"{req.side.upper()} {req.type.upper()} with qty {order_qty} @ limit $ {limit_price} & stop $ {stop_price}\n",
                    "  ",
                )
            )

            # for retry in range(3, 0, -1):
            try:
                # Submit order
                order: Order = self.trade_client.submit_order(req)
                f.write(
                    indent(
                        f"{order.status.upper()} with total qty {order.qty}, filled {order.filled_qty} at $ {order.filled_avg_price}\n",
                        "  ",
                    )
                )

                f.write(indent("OK\n\n", "  "))
                return OrderInfo(order)

            # except requests.exceptions.ConnectionError as e:
            #     print(red(f'  retry {retry}'), flush=True)
            #     if retry <= 1:
            #         raise e
            #     time.sleep(3.0 / retry)  # 1, 1.5, 3

            except APIError as e:
                print(f"Stop Limit order error: {e}")
                f.write(indent(f"Stop Limit error: {e}\n\n", "  "))
                raise e

    def replace_order(
        self,
        order: OrderInfo,
        # https://docs.alpaca.markets/reference/replaceorderforaccount-1
        # You can only patch FULL shares for now. Qty of equity fractional/notional orders are NOT allowed to change.
        # order_qty: float,
        limit_price: float,
        stop_price: float = 0.0,
    ) -> OrderInfo:
        stop_price = stop_price if stop_price else limit_price  # Ensure STOP price
        with open("martingale.audit.log", "a") as f:
            f.write(f"{datetime.now()} {order.side.upper()} {order.symbol}\n")

            # Refresh status
            order = self.get_order(order.id)

            # If partial/fill, return original order
            # https://docs.alpaca.markets/docs/orders-at-alpaca#order-lifecycle
            # PARTIALLY_FILLED -> FILLED
            # PARTIALLY_FILLED -> EXPIRED
            if (
                order.status == OrderStatus.FILLED
                or order.status == OrderStatus.PARTIALLY_FILLED
            ):
                return order

            # Construct STOP-LIMIT order
            if order.type == OrderType.STOP_LIMIT:
                req = ReplaceOrderRequest(
                    # order_qty=order_qty,
                    limit_price=limit_price,
                    stop_price=stop_price,
                )

            # Construct LIMIT order
            elif order.type == OrderType.LIMIT:
                req = ReplaceOrderRequest(
                    # order_qty=order_qty,
                    limit_price=limit_price,
                )

            f.write(
                indent(
                    f"REPLACE {order.side.upper()} {order.type.upper()} @ limit $ {limit_price} & stop $ {stop_price}\n",
                    "  ",
                )
            )
            # f.write(indent(f"REPLACE {order.side.upper()} {order.type.upper()} with qty {order_qty} @ limit $ {limit_price} & stop $ {stop_price or limit_price}\n", "  "))

            for retry in range(3, 0, -1):
                try:
                    # Submit order
                    new_order: Order = self.trade_client.replace_order_by_id(
                        order.id, req
                    )
                    f.write(
                        indent(
                            f"{order.status.upper()} with total qty {order.qty}, filled {order.filled_qty} at $ {order.filled_avg_price}\n",
                            "  ",
                        )
                    )

                    f.write(indent("OK\n\n", "  "))
                    return OrderInfo(new_order)

                except requests.exceptions.ConnectionError as e:
                    #  OK to retry since replacing an order (by id) is idempotent
                    print(red(f"  retry {retry}"), flush=True)
                    if retry <= 1:
                        raise e
                    time.sleep(3.0 / retry)  # 1, 1.5, 3

                except APIError as e:
                    # Tested replacing CANCELLED & FILLED orders, and both raise 40410000:
                    # API Error: {"code":40410000,"message":"order not found"}
                    print(f"Replace order error: {e}")
                    f.write(indent(f"Replace order error: {e}\n\n", "  "))
                    # raise e

            # Return original order if replace failed
            return order

    def cancel_renew_order(
        self,
        order: OrderInfo,
        order_qty: float,
        limit_price: float,
        stop_price: float = 0.0,
    ) -> OrderInfo:
        stop_price = stop_price if stop_price > 0 else limit_price  # Ensure STOP price
        with open("martingale.audit.log", "a") as f:
            f.write(f"{datetime.now()} {order.side.upper()} {order.symbol}\n")
            f.write(
                indent(
                    f"RENEW {order.side.upper()} {order.type.upper()} with qty {order_qty} @ limit $ {limit_price} & stop $ {stop_price or limit_price}\n",
                    "  ",
                )
            )

            # Refresh status
            order = self.get_order(order.id)

            # If partial/fill, return original order
            # https://docs.alpaca.markets/docs/orders-at-alpaca#order-lifecycle
            # PARTIALLY_FILLED -> FILLED
            # PARTIALLY_FILLED -> EXPIRED
            # PARTIALLY_FILLED -> PENDING_CANCEL -> CANCELED
            if (
                order.status == OrderStatus.FILLED
                or order.status == OrderStatus.PARTIALLY_FILLED
            ):
                return order

            # Try CANCEL old order
            if not self.cancel_order(order):
                # Else return original order (likely FILLED or PARTIALLY_FILLED)
                return order

            # Wait 5s for cancel to complete
            for i in range(10):
                order = self.get_order(order.id)
                if order.status == OrderStatus.CANCELED:
                    break
                print(f"    awaiting CANCEL {i + 1} ({order.status.upper()})")
                time.sleep(0.5)

            # If cancel still pending
            if order.status != OrderStatus.CANCELED:
                # Return original order (likely PENDING_CANCEL, FILLED or PARTIALLY_FILLED)
                return order

            # New STOP-LIMIT order
            if order.type == OrderType.STOP_LIMIT:
                order = self.submit_stop_limit_order(
                    order.side,
                    order_qty,  # Updated QTY
                    limit_price,
                    stop_price,
                )

            # New LIMIT order
            elif order.type == OrderType.LIMIT:
                order = self.submit_limit_order(
                    order.side,
                    order_qty,
                    limit_price,
                )

            return order

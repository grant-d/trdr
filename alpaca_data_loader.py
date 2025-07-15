import os
import pandas as pd
from datetime import datetime
from typing import Optional, Union
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

from base_data_loader import BaseDataLoader
from config_manager import Config
from timeframe import TimeFrame as CustomTimeFrame, TimeFrameUnit as CustomTimeFrameUnit

load_dotenv()


class AlpacaDataLoader(BaseDataLoader):
    """
    Alpaca-specific implementation of the data loader.
    
    This class handles data fetching from Alpaca Markets API, supporting both
    cryptocurrency and stock market data. It automatically routes requests to
    the appropriate API based on symbol type.
    
    Attributes:
        crypto_client: Client for fetching cryptocurrency data
        stock_client: Client for fetching stock market data
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the Alpaca data loader.
        
        Args:
            config: Configuration object with trading parameters
        """
        self.crypto_client: Optional[CryptoHistoricalDataClient] = None
        self.stock_client: Optional[StockHistoricalDataClient] = None
        super().__init__(config)

    def setup_clients(self) -> None:
        """
        Initialize Alpaca API clients for both crypto and stock data.
        
        Raises:
            ValueError: If API keys are not found in environment variables
        """
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not api_secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")

        # Crypto client
        self.crypto_client = CryptoHistoricalDataClient(api_key, api_secret)

        # Stock client
        self.stock_client = StockHistoricalDataClient(api_key, api_secret)

    def convert_to_alpaca_timeframe(self, timeframe_str: str) -> TimeFrame:
        """
        Convert our universal timeframe format to Alpaca's TimeFrame object.
        
        Args:
            timeframe_str: Timeframe string (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            Alpaca TimeFrame object
            
        Raises:
            ValueError: If timeframe cannot be converted to Alpaca format
        """
        # First parse using our custom TimeFrame
        custom_tf = CustomTimeFrame.from_string(timeframe_str)

        # Map our units to Alpaca units
        unit_mapping = {
            CustomTimeFrameUnit.MINUTE: TimeFrameUnit.Minute,
            CustomTimeFrameUnit.HOUR: TimeFrameUnit.Hour,
            CustomTimeFrameUnit.DAY: TimeFrameUnit.Day,
            CustomTimeFrameUnit.WEEK: TimeFrameUnit.Week,
        }

        alpaca_unit = unit_mapping.get(custom_tf.unit)
        if alpaca_unit is None:
            raise ValueError(f"Cannot convert timeframe unit: {custom_tf.unit}")
        # Ensure alpaca_unit is a TimeFrameUnit, not a string
        if isinstance(alpaca_unit, str):
            alpaca_unit = TimeFrameUnit[alpaca_unit]
        return TimeFrame(custom_tf.amount, alpaca_unit)

    def fetch_bars(self, start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch market data bars from Alpaca API.
        
        Automatically routes to crypto or stock API based on symbol type detected
        in the base class. Handles data conversion and adds calculated columns.
        
        Args:
            start: Start datetime for data range
            end: Optional end datetime (defaults to current UTC time)
            
        Returns:
            DataFrame with OHLCV data plus calculated hlc3 and dollar volume columns
            
        Raises:
            RuntimeError: If the appropriate client is not initialized
        """
        symbol = self.config.symbol
        timeframe = self.convert_to_alpaca_timeframe(self.config.timeframe)

        if end is None:
            end = datetime.utcnow()

        if self.is_crypto_symbol:
            if not self.crypto_client:
                raise RuntimeError("Crypto client not initialized")

            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            )
            bars = self.crypto_client.get_crypto_bars(request_params)
        else:
            if not self.stock_client:
                raise RuntimeError("Stock client not initialized")

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            )
            bars = self.stock_client.get_stock_bars(request_params)

        # Convert to DataFrame
        df_list: list[dict[str, Union[datetime, float, int]]] = []

        # Alpaca returns a BarSet object
        # When we request a single symbol, we can access it directly
        if hasattr(bars, symbol):
            # Access bars for the specific symbol
            symbol_bars = getattr(bars, symbol, [])
        elif hasattr(bars, '__getitem__'):
            try:
                symbol_bars = bars[symbol]
            except (KeyError, TypeError):
                symbol_bars = []
        else:
            symbol_bars = []

        # Now iterate over the bars for our symbol
        for bar in symbol_bars:
            df_list.append({
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume),
                'trade_count': int(bar.trade_count) if hasattr(bar, 'trade_count') and bar.trade_count is not None else 0,
            })

        if not df_list:
            return pd.DataFrame()

        df = pd.DataFrame(df_list)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Add calculated columns
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['dv'] = df['hlc3'] * df['volume']

        return df

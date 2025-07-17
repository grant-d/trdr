import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from abc import ABC, abstractmethod

from config_manager import Config
from filename_utils import generate_filename, get_data_path
from timeframe import TimeFrame


class BaseDataLoader(ABC):
    """
    Abstract base class for market data loaders.

    This class provides the common functionality for loading market data from various sources,
    managing CSV storage, and handling data imputation for missing bars. Subclasses must
    implement the specific data source connection and fetching logic.

    Attributes:
        config: Configuration object containing symbol, timeframe, and other settings
        is_crypto_symbol: Boolean indicating if the symbol is a cryptocurrency
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the base data loader.

        Args:
            config: Configuration object with trading parameters
        """
        self.config = config
        self.is_crypto_symbol = self._detect_crypto_symbol(config.symbol)
        self.setup_clients()

    def _detect_crypto_symbol(self, symbol: str) -> bool:
        """
        Detect if a symbol represents a cryptocurrency based on common patterns.

        Args:
            symbol: The trading symbol to analyze

        Returns:
            True if the symbol appears to be a cryptocurrency, False otherwise

        Examples:
            - "BTC/USD" -> True (contains slash)
            - "BTCUSD" -> True (ends with USD)
            - "ETHUSDT" -> True (ends with USDT)
            - "AAPL" -> False (stock symbol)
        """
        symbol_upper = symbol.upper()

        # Common crypto patterns
        crypto_patterns = [
            # Slash pairs (BTC/USD, ETH/USDT)
            "/" in symbol,
            # Crypto suffixes
            symbol_upper.endswith("USD"),
            symbol_upper.endswith("USDT"),
            symbol_upper.endswith("USDC"),
            symbol_upper.endswith("EUR"),
            symbol_upper.endswith("GBP"),
            symbol_upper.endswith("BTC"),
            symbol_upper.endswith("ETH"),
            # Common crypto symbols
            symbol_upper.startswith("BTC"),
            symbol_upper.startswith("ETH"),
            symbol_upper.startswith("XRP"),
            symbol_upper.startswith("ADA"),
            symbol_upper.startswith("DOT"),
            symbol_upper.startswith("DOGE"),
            symbol_upper.startswith("SOL"),
            symbol_upper.startswith("MATIC"),
            symbol_upper.startswith("AVAX"),
            symbol_upper.startswith("LINK"),
        ]

        return any(crypto_patterns)

    @abstractmethod
    def setup_clients(self) -> None:
        """
        Initialize API clients for the specific data source.

        This method must be implemented by subclasses to set up their specific
        API connections and authentication.
        """
        pass

    @abstractmethod
    def fetch_bars(
        self, start: datetime, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch market data bars from the data source.

        This method must be implemented by subclasses to fetch data from their
        specific data source (e.g., Alpaca, Binance, etc.).

        Args:
            start: Start datetime for the data range
            end: Optional end datetime (defaults to current time)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume,
            trade_count, hlc3, dv
        """
        pass

    def get_csv_path(self) -> str:
        """
        Generate the full path for the CSV file based on current configuration.

        Returns:
            Full path to the CSV file (e.g., "data/btc_usd_1m_bars.csv")
        """
        filename = generate_filename(
            self.config.symbol, self.config.timeframe, "bars", "csv"
        )
        return get_data_path(filename)

    def load_existing_data(self) -> Optional[pd.DataFrame]:
        """
        Load existing market data from CSV file if it exists.

        Returns:
            DataFrame with historical data if file exists, None otherwise
        """
        csv_path = self.get_csv_path()
        if os.path.exists(csv_path):
            # Check if file is empty or has no content
            if os.path.getsize(csv_path) == 0:
                return None
            try:
                df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                # Check if dataframe is empty after reading
                if df.empty:
                    return None
                return df
            except pd.errors.EmptyDataError:
                # File exists but has no columns/data to parse
                return None
        return None

    def load_data(self) -> pd.DataFrame:
        """
        Main method to load market data with intelligent caching.

        This method handles both initial bulk loading and incremental updates:
        - If no existing data: Performs bulk load for min_bars periods
        - If data exists: Performs catchup load from last timestamp

        The method also handles data imputation, CSV persistence, and state updates.

        Returns:
            Complete DataFrame with all historical and current data

        Side effects:
            - Saves data to CSV file
            - Updates config state with total bars and last sync time
        """
        existing_df = self.load_existing_data()

        # Initial bulk load
        if existing_df is None or existing_df.empty:
            print(
                f"Performing initial bulk load for {self.config.symbol} {self.config.timeframe}"
            )

            # Calculate start date based on min_bars
            timeframe = TimeFrame.from_string(self.config.timeframe)
            minutes = timeframe.to_minutes()
            start = datetime.utcnow() - timedelta(
                minutes=minutes * self.config.min_bars
            )

            df = self.fetch_bars(start)

        # Catchup load
        else:
            last_timestamp = existing_df["timestamp"].max()
            print(f"Performing catchup load from {last_timestamp}")

            # Fetch new bars
            new_df = self.fetch_bars(last_timestamp + timedelta(seconds=1))

            if not new_df.empty:
                # Combine with existing data
                df = pd.concat([existing_df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"], keep="last")
                df = df.sort_values("timestamp").reset_index(drop=True)
            else:
                df = existing_df

        # TODO: Clean data

        # Save to CSV
        csv_path = self.get_csv_path()
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} bars to {csv_path}")

        return df

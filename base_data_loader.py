import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod
import warnings

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

    def load_data(self, clean_data: bool = True) -> pd.DataFrame:
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

        # Clean data
        if clean_data and not df.empty:
            df = self.clean_data(df)

        # Save to CSV
        csv_path = self.get_csv_path()
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} bars to {csv_path}")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline for financial market data.

        This method applies multiple cleaning techniques in sequence:
        1. Validate basic data structure and required columns
        2. Handle missing values with forward fill
        3. Detect and handle outliers using multiple methods
        4. Validate OHLCV data integrity
        5. Recalculate derived fields (hlc3, dv)

        Args:
            df: DataFrame with market data columns

        Returns:
            Cleaned DataFrame with validated and processed data
        """
        if df.empty:
            return df

        # Store original row count for reporting
        original_count = len(df)

        # 1. Validate data structure
        df = self._validate_data_structure(df)

        # 2. Handle missing values
        df = self._handle_missing_values(df)

        # 3. Detect and handle outliers
        df = self._detect_and_handle_outliers(df)

        # 4. Validate OHLCV integrity
        df = self._validate_ohlcv_integrity(df)

        # 5. Recalculate derived fields
        df = self._recalculate_derived_fields(df)

        # 6. Final validation and cleanup
        df = self._final_cleanup(df)

        cleaned_count = len(df)
        if cleaned_count != original_count:
            print(
                f"Data cleaning: {original_count} → {cleaned_count} bars ({original_count - cleaned_count} removed)"
            )

        return df

    def _validate_data_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that the DataFrame has the required columns and correct data types.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with validated structure
        """
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure numeric columns are numeric
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle trade_count column
        if "trade_count" in df.columns:
            df["trade_count"] = pd.to_numeric(
                df["trade_count"], errors="coerce"
            ).fillna(0)
        else:
            df["trade_count"] = 0

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in financial data using forward fill and validation.

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled
        """
        # Count missing values before processing
        missing_before = df.isnull().sum().sum()

        # Forward fill missing values for price columns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Handle volume - use 0 for missing volume
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        # Handle trade_count - use 0 for missing trade count
        if "trade_count" in df.columns:
            df["trade_count"] = df["trade_count"].fillna(0)

        # Remove rows where all price columns are still NaN (beginning of series)
        df = df.dropna(subset=price_cols, how="all")

        missing_after = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"Missing values handled: {missing_before} → {missing_after}")

        return df

    def _detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using multiple statistical methods tailored for financial data.

        Methods used:
        1. Z-score method for price columns (threshold: 3.0)
        2. IQR method for volume (threshold: 1.5)
        3. Financial-specific rules (price jumps, volume spikes)

        Args:
            df: DataFrame with market data

        Returns:
            DataFrame with outliers handled
        """
        outliers_detected = 0

        # 1. Z-score method for price columns
        price_cols = ["open", "high", "low", "close"]
        z_threshold = 3.0

        for col in price_cols:
            if (
                col in df.columns and len(df) > 10
            ):  # Need sufficient data for statistics
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > z_threshold

                if outliers.any():
                    outliers_detected += outliers.sum()
                    # Cap outliers at percentile boundaries rather than removing
                    upper_bound = df[col].quantile(0.99)
                    lower_bound = df[col].quantile(0.01)
                    df.loc[outliers, col] = np.clip(
                        pd.Series(df.loc[outliers, col]), lower_bound, upper_bound
                    )

        # 2. IQR method for volume (exclude zero volumes for better statistics)
        if "volume" in df.columns and len(df) > 10:
            # Filter out zero volumes for statistical calculations
            non_zero_volumes = df[df["volume"] > 0]["volume"]
            
            if len(non_zero_volumes) > 5:  # Need some non-zero volumes for statistics
                Q1 = non_zero_volumes.quantile(0.25)
                Q3 = non_zero_volumes.quantile(0.75)
                IQR = Q3 - Q1
                
                # Only flag as outliers if IQR is meaningful (not too small)
                if IQR > 0:
                    # More conservative threshold for volume (3.0 instead of 2.0)
                    volume_outliers = df["volume"] > (Q3 + 3.0 * IQR)
                    
                    if volume_outliers.any():
                        outliers_detected += volume_outliers.sum()
                        # Cap volume outliers at 99.5th percentile of non-zero volumes
                        volume_cap = non_zero_volumes.quantile(0.995)
                        df.loc[volume_outliers, "volume"] = volume_cap

        # 3. Financial-specific rules
        outliers_detected += self._apply_financial_outlier_rules(df)

        if outliers_detected > 0:
            print(f"Outliers detected and handled: {outliers_detected}")

        return df

    def _apply_financial_outlier_rules(self, df: pd.DataFrame) -> int:
        """
        Apply financial market-specific outlier detection rules.

        Args:
            df: DataFrame with market data

        Returns:
            Number of outliers detected and handled
        """
        outliers_handled = 0

        if len(df) < 2:
            return outliers_handled

        # Rule 1: Detect extreme price jumps (>20% change)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                price_changes = df[col].pct_change().abs()
                extreme_jumps = price_changes > 0.20  # 20% threshold

                if extreme_jumps.any():
                    # Use previous value for extreme jumps
                    for idx in df.index[extreme_jumps]:
                        if idx > 0:
                            df.loc[idx, col] = df.loc[idx - 1, col]
                            outliers_handled += 1

        # Rule 2: Volume spikes (exclude zero volumes from median calculation)
        if "volume" in df.columns and len(df) > 10:
            # Calculate rolling median of non-zero volumes only
            non_zero_mask = df["volume"] > 0
            if non_zero_mask.sum() > 5:  # Need some non-zero volumes
                # Calculate median of non-zero volumes in recent window
                recent_non_zero_volumes = df[non_zero_mask]["volume"].tail(20)
                if len(recent_non_zero_volumes) > 0:
                    median_non_zero_volume = recent_non_zero_volumes.median()
                    
                    # Only flag volumes that are extremely high (>50x median non-zero volume)
                    volume_spikes = df["volume"] > (median_non_zero_volume * 50)
                    
                    if volume_spikes.any():
                        # Cap volume spikes at 10x median non-zero volume
                        for idx in df.index[volume_spikes]:
                            df.loc[idx, "volume"] = median_non_zero_volume * 10
                            outliers_handled += 1

        return outliers_handled

    def _validate_ohlcv_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix OHLCV data integrity issues.

        Checks and fixes:
        1. High >= Low
        2. Open and Close within High/Low range
        3. Volume >= 0
        4. Price values > 0

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with integrity issues fixed
        """
        fixes_applied = 0

        # Check 1: High >= Low
        invalid_hl = df["high"] < df["low"]
        if invalid_hl.any():
            # Swap high and low values
            df.loc[invalid_hl, ["high", "low"]] = df.loc[
                invalid_hl, ["low", "high"]
            ].values
            fixes_applied += invalid_hl.sum()

        # Check 2: Open within High/Low range
        invalid_open = (df["open"] > df["high"]) | (df["open"] < df["low"])
        if invalid_open.any():
            # Clamp open to high/low range
            df.loc[invalid_open, "open"] = np.clip(
                df.loc[invalid_open, "open"],
                df.loc[invalid_open, "low"],
                df.loc[invalid_open, "high"],
            )
            fixes_applied += invalid_open.sum()

        # Check 3: Close within High/Low range
        invalid_close = (df["close"] > df["high"]) | (df["close"] < df["low"])
        if invalid_close.any():
            # Clamp close to high/low range
            df.loc[invalid_close, "close"] = np.clip(
                df.loc[invalid_close, "close"],
                df.loc[invalid_close, "low"],
                df.loc[invalid_close, "high"],
            )
            fixes_applied += invalid_close.sum()

        # Check 4: Volume >= 0
        invalid_volume = df["volume"] < 0
        if invalid_volume.any():
            df.loc[invalid_volume, "volume"] = 0
            fixes_applied += invalid_volume.sum()

        # Check 5: Price values > 0
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            invalid_prices = df[col] <= 0
            if invalid_prices.any():
                # Replace invalid prices with NaN, then forward fill
                df.loc[invalid_prices, col] = np.nan
                df[col] = df[col].ffill()
                fixes_applied += invalid_prices.sum()

        if fixes_applied > 0:
            print(f"OHLCV integrity issues fixed: {fixes_applied}")

        return df

    def _recalculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalculate derived fields after data cleaning.

        Args:
            df: DataFrame with cleaned OHLCV data

        Returns:
            DataFrame with recalculated derived fields
        """
        # Recalculate hlc3 (High, Low, Close average)
        df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3

        # Recalculate dollar volume (dv)
        df["dv"] = df["hlc3"] * df["volume"]

        return df

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final cleanup and validation steps.

        Args:
            df: DataFrame after all cleaning steps

        Returns:
            Final cleaned DataFrame
        """
        # Sort by timestamp to ensure chronological order
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Remove any remaining rows with NaN in critical columns
        critical_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=critical_cols)

        # Ensure trade_count is integer
        if "trade_count" in df.columns:
            df["trade_count"] = df["trade_count"].astype(int)

        return df

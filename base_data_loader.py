import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
from abc import ABC, abstractmethod
import warnings
from tsfracdiff import FractionalDifferentiator
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

    def get_csv_path(self, suffix: str) -> str:
        """
        Generate the full path for the CSV file based on current configuration.

        Args:
            suffix: Optional suffix to add before .csv (e.g., "clean", "transform")
                   If None, returns the base CSV path

        Returns:
            Full path to the CSV file
            Examples:
            - suffix=None: "data/btc_usd_1m_bars.csv"
            - suffix="clean": "data/btc_usd_1m_bars.clean.csv"
            - suffix="transform": "data/btc_usd_1m_bars.transform.csv"
        """
        base_name = f"{self.config.symbol}_{self.config.timeframe}"
        base_name = (
            base_name.replace("/", "_").replace("-", "_").replace(":", "_").lower()
        )  # BTC/USD -> btc_usd

        filename = f"{base_name}.{suffix}.csv"

        return get_data_path(filename)

    def load_existing_data(self) -> Optional[pd.DataFrame]:
        """
        Load existing market data from CSV file if it exists.

        Returns:
            DataFrame with historical data if file exists, None otherwise
        """
        csv_path = self.get_csv_path("bars")
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

    def load_data(self, stage_data: bool = True) -> pd.DataFrame:
        """
        Main method to load market data with intelligent caching.

        This method handles both initial bulk loading and incremental updates:
        - If no existing data: Performs bulk load for min_bars periods
        - If data exists: Performs catchup load from last timestamp

        Args:
            stage_data: If True, save data to CSV file. Default True

        Returns:
            Complete DataFrame with all historical and current data (raw, not cleaned or transformed).

        Side effects:
            - Saves data to CSV file (if stage_data is True)
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

        # Save to CSV if requested
        if stage_data:
            csv_path = self.get_csv_path("bars")
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(df)} bars to {csv_path}")

        return df

    def transform(
        self,
        df: pd.DataFrame,
        frac_diff: Union[bool, str, None] = None,
        log_volume: Union[bool, str, None] = None,
        stage_data: bool = True,
    ) -> pd.DataFrame:
        """
        Apply transformations to the data (fractional differentiation, log volume).

        Args:
            df: DataFrame with cleaned market data
            frac_diff: How to apply fractional differentiation. Options:
                           - None/False: Don't apply fractional differentiation
                           - True: Replace original columns with differentiated values
                           - str: Add new columns with the string as suffix (e.g., '_fd')
            log_volume: How to apply log transformation to volume columns. Options:
                           - None/False: Don't apply log transformation
                           - True: Replace original volume columns with log values
                           - str: Add new columns with the string as suffix (e.g., '_lr')
            stage_data: If True, save transformed data to .clean.csv file

        Returns:
            DataFrame with requested transformations applied
        """
        if df.empty:
            return df

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Apply fractional differentiation if requested
        if frac_diff:
            if isinstance(frac_diff, bool):
                # True means overwrite
                result_df, orders = self.frac_diff(result_df, overwrite=True)
            elif isinstance(frac_diff, str):
                # String means add columns with that suffix
                df_with_fd, orders = self.frac_diff(result_df, overwrite=False)

                # The frac_diff method always creates columns with '_fd' suffix by default
                # If user wants a different suffix, we need to rename those columns
                if frac_diff != "_fd":
                    for col in ["open", "high", "low", "close", "hlc3", "dv"]:
                        if f"{col}_fd" in df_with_fd.columns:
                            df_with_fd = df_with_fd.rename(
                                columns={f"{col}_fd": f"{col}{frac_diff}"}
                            )
                result_df = df_with_fd
            else:
                raise ValueError(
                    f"Invalid frac_diff type: {type(frac_diff)}. "
                    "Must be bool, str, or None"
                )

        # Apply log transformation to volume if requested
        if log_volume:
            if isinstance(log_volume, bool):
                # True means overwrite
                result_df = self.log_transform_volume(result_df, overwrite=True)
            elif isinstance(log_volume, str):
                # String means add columns with that suffix
                df_with_log = self.log_transform_volume(result_df, overwrite=False)

                # The log_transform_volume method always creates columns with '_log' suffix by default
                # If user wants a different suffix, we need to rename those columns
                if log_volume != "_log":
                    if "volume_log" in df_with_log.columns:
                        df_with_log = df_with_log.rename(
                            columns={"volume_log": f"volume{log_volume}"}
                        )
                    if "dv_log" in df_with_log.columns:
                        df_with_log = df_with_log.rename(
                            columns={"dv_log": f"dv{log_volume}"}
                        )
                result_df = df_with_log
            else:
                raise ValueError(
                    f"Invalid log_volume type: {type(log_volume)}. "
                    "Must be bool, str, or None"
                )

        # Save transformed data to .transform.csv if requested
        if stage_data and (frac_diff or log_volume):
            transform_csv_path = self.get_csv_path("transform")
            result_df.to_csv(transform_csv_path, index=False)
            print(f"Saved transformed data to {transform_csv_path}")

        return result_df

    def clean_data(self, df: pd.DataFrame, stage_data: bool = True) -> pd.DataFrame:
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
            stage_data: If True, save cleaned data to .clean.csv file

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

        # 7. Final NaN handling - ensure no NaN values remain
        # Fill any remaining NaN values in price columns with forward fill
        price_cols = ["open", "high", "low", "close", "hlc3"]
        for col in price_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill()
                # If still NaN (e.g., first row), use backward fill
                if df[col].isna().any():
                    df[col] = df[col].bfill()

        # Recalculate derived fields after filling
        if any(
            col in df.columns and df[col].isna().any()
            for col in ["high", "low", "close"]
        ):
            df = self._recalculate_derived_fields(df)

        cleaned_count = len(df)
        if cleaned_count != original_count:
            print(
                f"Data cleaning: {original_count} → {cleaned_count} bars ({original_count - cleaned_count} removed)"
            )

        # Save cleaned data to .clean.csv if requested
        if stage_data:
            clean_csv_path = self.get_csv_path("clean")
            df.to_csv(clean_csv_path, index=False)
            print(f"Saved cleaned data to {clean_csv_path}")

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
                # Calculate z-scores, handling edge cases
                std_dev = df[col].std()
                if std_dev == 0 or pd.isna(std_dev):
                    continue  # Skip if no variation in data
                z_scores = np.abs((df[col] - df[col].mean()) / std_dev)
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
                # Skip first row which will be NaN from pct_change
                extreme_jumps = (price_changes > 0.20) & price_changes.notna()

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

    def frac_diff(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        overwrite: bool = False,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Apply fractional differentiation to specified columns.

        Uses tsfracdiff with default settings to automatically determine optimal
        differentiation orders to achieve stationarity. The transformation typically
        results in NaN values for the first few rows due to lag requirements.

        Args:
            df: DataFrame with time series data
            columns: List of column names to differentiate. If None, applies to
                    ['open', 'high', 'low', 'close', 'hlc3', 'dv']
            overwrite: If True, replace original columns with differentiated values.
                      If False, create new columns with '_fd' suffix. Default False
            drop_na: If True, drop rows with NaN values in the differentiated columns.
                    Default True

        Returns:
            Tuple of (DataFrame, Dict[str, float]):
            - DataFrame with fractionally differentiated values. If overwrite=False,
              new columns named as '{original_column}_fd' are added. If overwrite=True,
              original columns are replaced. Rows with NaN are dropped if drop_na=True.
            - Dictionary mapping column names to their fractional differentiation orders

        Example:
            # Apply fractional differentiation
            df_enhanced, orders = loader.frac_diff(df, columns=['close', 'open'])
            # orders = {'close': 0.456, 'open': 0.523}

            # Overwrite original columns
            df_overwrite, orders = loader.frac_diff(df, columns=['close', 'dv'], overwrite=True)
        """
        # Default columns if none specified
        if columns is None:
            columns = ["open", "high", "low", "close"]
            # Add hlc3 & dv if exist
            if "hlc3" in df.columns:
                columns.append("hlc3")
            if "dv" in df.columns:
                columns.append("dv")

        # Filter to only columns that exist in the dataframe
        columns_to_process = [col for col in columns if col in df.columns]

        if not columns_to_process:
            warnings.warn("No specified columns found in DataFrame")
            return df, {}

        # Create a copy to avoid modifying original
        result_df = df.copy()

        # Dictionary to store fractional orders
        orders_dict: Dict[str, float] = {}

        # Process each column
        for col in columns_to_process:
            try:
                # Extract series and handle any NaN values
                series = df[[col]].dropna()

                if len(series) < 100:  # tsfracdiff needs reasonable data length
                    warnings.warn(
                        f"Column '{col}' has insufficient data ({len(series)} rows). "
                        f"Skipping fractional differentiation."
                    )
                    continue

                # Initialize fractional differentiator with empty constructor
                differentiator = FractionalDifferentiator()

                # Auto-fit and transform
                # Suppress numpy FutureWarning about DataFrame.swapaxes from tsfracdiff
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message="'DataFrame.swapaxes' is deprecated",
                    )
                    transformed = differentiator.FitTransform(series)

                # Get the estimated order
                orders = (
                    list(differentiator.orders)
                    if differentiator.orders is not None
                    else []
                )
                order = float(orders[0]) if orders and orders[0] is not None else 0.0

                # print(f"Auto-fitted fractional order for '{col}': {order:.4f}")

                # Handle the transformed data
                if isinstance(transformed, pd.DataFrame):
                    frac_diff_values = transformed.iloc[:, 0].values
                else:
                    frac_diff_values = transformed.flatten()

                # Determine column names based on overwrite setting
                if overwrite:
                    target_col_name = col
                else:
                    target_col_name = f"{col}_fd"

                # Get the original indices for alignment
                original_indices = series.index

                # Align the differentiated values with original dataframe
                # tsfracdiff may return fewer values due to lag requirements
                if len(frac_diff_values) < len(original_indices):
                    # Calculate how many values were lost
                    values_lost = len(original_indices) - len(frac_diff_values)
                    # Use the last indices (most recent data)
                    valid_indices = original_indices[values_lost:]
                    result_df.loc[valid_indices, target_col_name] = frac_diff_values
                else:
                    result_df.loc[original_indices, target_col_name] = frac_diff_values

                # Store the order used in the dictionary
                orders_dict[col] = float(order)

                # print(f"Created fractionally differentiated column: '{target_col_name}'")

            except Exception as e:
                warnings.warn(
                    f"Error applying fractional differentiation to column '{col}': {str(e)}"
                )
                continue

        # Drop rows with NaN values if requested
        if drop_na and columns_to_process:
            # Get all columns that were created/modified
            columns_to_check = []
            if overwrite:
                columns_to_check = columns_to_process
            else:
                columns_to_check = [f"{col}_fd" for col in columns_to_process]

            # Filter to columns that exist in the result
            columns_to_check = [
                col for col in columns_to_check if col in result_df.columns
            ]

            if columns_to_check:
                # rows_before = len(result_df)
                result_df = result_df.dropna(subset=columns_to_check)
                # rows_after = len(result_df)
                # if rows_before != rows_after:
                #     print(f"Fractional differentiation: dropped {rows_before - rows_after} rows with NaN values")

        return result_df, orders_dict

    def log_transform_volume(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        overwrite: bool = False,
        epsilon: float = 1e-8,
        drop_first: bool = True,
    ) -> pd.DataFrame:
        """
        Apply log transformation to volume-related columns.

        Log transformation is useful for volume data to:
        - Handle the wide range of values
        - Reduce impact of extreme outliers
        - Make the distribution more normal-like
        - Transform multiplicative processes to additive

        Note: The first row represents an absolute value rather than a relative
        change, so it is dropped by default for consistency in time series analysis.

        Args:
            df: DataFrame with volume data
            columns: List of column names to transform. If None, applies to
                    ['volume'] only
            overwrite: If True, replace original columns with log-transformed values.
                      If False, create new columns with '_log' suffix. Default False
            epsilon: Small value added to avoid log(0). Default 1e-8
            drop_first: If True, drop the first row after transformation. Default True

        Returns:
            DataFrame with log-transformed volume columns. First row is dropped
            if drop_first=True.

        Example:
            # Apply log transformation
            df = loader.log_transform_volume(df)
            # Creates 'volume_log' column, drops first row

            # Overwrite original columns
            df = loader.log_transform_volume(df, overwrite=True)
        """
        # Default columns if none specified
        if columns is None:
            columns = []
            if "volume" in df.columns:
                columns.append("volume")

        # Filter to only columns that exist in the dataframe
        columns_to_process = [col for col in columns if col in df.columns]

        if not columns_to_process:
            warnings.warn("No specified volume columns found in DataFrame")
            return df

        # Create a copy to avoid modifying original
        result_df = df.copy()

        # Process each column
        for col in columns_to_process:
            try:
                # Add epsilon to avoid log(0)
                safe_values = result_df[col] + epsilon

                # Apply log transformation
                log_values = np.log(safe_values)

                # Determine column name based on overwrite setting
                if overwrite:
                    target_col_name = col
                else:
                    target_col_name = f"{col}_log"

                # Store the log-transformed values
                result_df[target_col_name] = log_values

                # print(f"Applied log transformation to '{col}' -> '{target_col_name}'")

            except Exception as e:
                warnings.warn(
                    f"Error applying log transformation to column '{col}': {str(e)}"
                )
                continue

        # Drop first row if requested
        if drop_first and len(result_df) > 1:
            result_df = result_df.iloc[1:].reset_index(drop=True)
            print("Log transformation: dropped first row (absolute value)")

        return result_df

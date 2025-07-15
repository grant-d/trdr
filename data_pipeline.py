"""
Data pipeline for cleaning and transforming market data.

This module provides a DataPipeline class that handles data cleaning,
validation, and transformation operations including conversion to
alternative bar types like dollar bars.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, Literal
from pathlib import Path
from scipy import stats

from bar_aggregators import DollarBarAggregator, VolumeBarAggregator


class DataPipeline:
    """
    Pipeline for processing market data through cleaning and transformation steps.
    
    Handles data quality issues like missing values, extreme outliers,
    and provides conversion to alternative bar formats.
    """
    
    def __init__(
        self,
        outlier_std_threshold: float = 5.0,
        price_columns: list[str] | None = None,
        volume_column: str = "volume"
    ) -> None:
        """
        Initialize data pipeline with configuration.
        
        Args:
            outlier_std_threshold: Number of standard deviations for outlier detection
            price_columns: List of price columns to check for outliers (default: OHLC)
            volume_column: Name of volume column
        """
        self.outlier_std_threshold = outlier_std_threshold
        self.price_columns = price_columns or ["open", "high", "low", "close"]
        self.volume_column = volume_column
        self._processing_stats = {}
        
    def load_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
        
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in data columns using appropriate strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values filled
        """
        df_filled = df.copy()
        filled_count = 0
        
        # Forward fill price columns (carry previous value forward)
        for col in self.price_columns:
            if col in df_filled.columns:
                na_count = df_filled[col].isna().sum()
                if na_count > 0:
                    df_filled[col] = df_filled[col].ffill()
                    # If still NaN at beginning, backfill
                    df_filled[col] = df_filled[col].bfill()
                    filled_count += na_count
                    print(f"Filled {na_count} missing values in {col}")
        
        # Fill volume with 0 (no trades)
        if self.volume_column in df_filled.columns:
            na_count = df_filled[self.volume_column].isna().sum()
            if na_count > 0:
                df_filled[self.volume_column] = df_filled[self.volume_column].fillna(0)
                filled_count += na_count
                print(f"Filled {na_count} missing values in {self.volume_column} with 0")
        
        # Fill trade_count with 0
        if 'trade_count' in df_filled.columns:
            na_count = df_filled['trade_count'].isna().sum()
            if na_count > 0:
                df_filled['trade_count'] = df_filled['trade_count'].fillna(0)
                filled_count += na_count
                
        self._processing_stats['filled_missing'] = filled_count
        
        if filled_count > 0:
            print(f"Total filled values: {filled_count}")
            
        return df_filled
        
    def clamp_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clamp extreme outlier values using statistical thresholds.
        
        Uses rolling window statistics to detect and cap outliers while
        preserving legitimate price movements.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extreme values clamped
        """
        df_clean = df.copy()
        clamped_count = 0
        
        for col in self.price_columns:
            if col not in df_clean.columns:
                continue
                
            # Calculate rolling statistics for adaptive thresholds
            window_size = min(100, len(df_clean) // 10)
            if window_size < 10:
                window_size = len(df_clean)
                
            rolling_mean = df_clean[col].rolling(window=window_size, center=True).mean()
            rolling_std = df_clean[col].rolling(window=window_size, center=True).std()
            
            # Fill edge cases
            rolling_mean = rolling_mean.fillna(df_clean[col].mean())
            rolling_std = rolling_std.fillna(df_clean[col].std())
            
            # Calculate thresholds
            upper_threshold = rolling_mean + (self.outlier_std_threshold * rolling_std)
            lower_threshold = rolling_mean - (self.outlier_std_threshold * rolling_std)
            
            # Clamp values
            original_values = df_clean[col].copy()
            df_clean[col] = df_clean[col].clip(lower=lower_threshold, upper=upper_threshold)
            
            # Count clamped values
            clamped = (original_values != df_clean[col]).sum()
            clamped_count += clamped
            
            if clamped > 0:
                print(f"Clamped {clamped} extreme values in {col}")
                
        self._processing_stats['clamped_values'] = clamped_count
        
        return df_clean
        
    def validate_price_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure logical price relationships (high >= low, etc).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected price relationships
        """
        df_clean = df.copy()
        corrections = 0
        
        # Ensure high >= low
        mask = df_clean['high'] < df_clean['low']
        if mask.any():
            # Swap high and low where relationship is violated
            df_clean.loc[mask, ['high', 'low']] = df_clean.loc[mask, ['low', 'high']].values
            corrections += mask.sum()
            
        # Ensure high >= open and high >= close
        df_clean['high'] = df_clean[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low <= open and low <= close
        df_clean['low'] = df_clean[['low', 'open', 'close']].min(axis=1)
        
        if corrections > 0:
            print(f"Corrected {corrections} price relationship violations")
            
        self._processing_stats['price_corrections'] = corrections
        
        return df_clean
        
    def remove_zero_volume_bars(self, df: pd.DataFrame, keep_percentage: float = 0.1) -> pd.DataFrame:
        """
        Remove excessive zero-volume bars while keeping some for continuity.
        
        Args:
            df: Input DataFrame
            keep_percentage: Percentage of zero-volume bars to randomly keep
            
        Returns:
            DataFrame with reduced zero-volume bars
        """
        zero_volume_mask = df[self.volume_column] == 0
        zero_volume_count = zero_volume_mask.sum()
        
        if zero_volume_count == 0:
            return df
            
        # Keep all non-zero volume bars
        df_clean = df[~zero_volume_mask].copy()
        
        # Randomly sample some zero-volume bars to keep
        zero_volume_df = df[zero_volume_mask]
        keep_count = int(zero_volume_count * keep_percentage)
        
        if keep_count > 0:
            kept_zeros = zero_volume_df.sample(n=min(keep_count, len(zero_volume_df)))
            df_clean = pd.concat([df_clean, kept_zeros]).sort_values('timestamp').reset_index(drop=True)
            
        removed = zero_volume_count - keep_count
        self._processing_stats['removed_zero_volume'] = removed
        
        if removed > 0:
            print(f"Removed {removed} zero-volume bars (kept {keep_count})")
            
        return df_clean
        
    def add_calculated_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add useful calculated columns if not present.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional calculated columns
        """
        # Add HLC3 (typical price) if not present
        if 'hlc3' not in df.columns:
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            
        # Add dollar volume if not present
        if 'dv' not in df.columns and self.volume_column in df.columns:
            df['dv'] = df['hlc3'] * df[self.volume_column]
            
        return df
        
    def process(
        self,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[Union[str, Path]] = None,
        zero_volume_keep_percentage: float = 0.1,
        dollar_bar_threshold: Optional[float] = None,
        price_column: str = "close"
    ) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Process data through the full pipeline.
        
        Args:
            df: Input DataFrame (if None, must provide csv_path)
            csv_path: Path to CSV file (if df is None)
            zero_volume_keep_percentage: Percentage of zero-volume bars to keep (0.0 = keep all, 1.0 = keep none)
            dollar_bar_threshold: If provided, convert to dollar bars
            price_column: Column to use for dollar bar calculation
            
        Returns:
            Tuple of (cleaned_df, dollar_bars_df or None)
        """
        # Reset stats
        self._processing_stats = {}
        
        # Load data if needed
        if df is None:
            if csv_path is None:
                raise ValueError("Must provide either df or csv_path")
            df = self.load_csv(csv_path)
            
        print(f"\nStarting pipeline with {len(df)} rows")
        
        # Processing steps
        df_clean = self.fill_missing_values(df)
        df_clean = self.clamp_extreme_values(df_clean)
        df_clean = self.validate_price_relationships(df_clean)
        
        # Remove zero-volume bars based on keep percentage
        # 0.0 = keep none (remove all), 1.0 = keep all (remove none)
        if zero_volume_keep_percentage < 1.0:
            df_clean = self.remove_zero_volume_bars(df_clean, keep_percentage=zero_volume_keep_percentage)
            
        df_clean = self.add_calculated_columns(df_clean)
        
        print(f"Pipeline complete: {len(df_clean)} rows remaining")
        
        # Convert to dollar bars if requested
        dollar_bars_df = None
        if dollar_bar_threshold is not None:
            print(f"\nConverting to dollar bars with threshold: ${dollar_bar_threshold:,.2f}")
            aggregator = DollarBarAggregator(
                threshold=dollar_bar_threshold,
                price_column=price_column,
                volume_column=self.volume_column
            )
            dollar_bars_df = aggregator.aggregate(df_clean)
            print(f"Created {len(dollar_bars_df)} dollar bars")
            
        return df_clean, dollar_bars_df
        
    def get_processing_stats(self) -> dict:
        """
        Get statistics from the last processing run.
        
        Returns:
            Dictionary with processing statistics
        """
        return self._processing_stats.copy()
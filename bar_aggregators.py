import math
import pandas as pd
import numpy as np
from typing import Optional, Literal, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod


class BaseBarAggregator(ABC):
    """
    Abstract base class for alternative bar aggregation methods.
    
    Provides common functionality for aggregating time-based bars into
    alternative bar types (dollar bars, volume bars, tick bars, etc.)
    """
    
    def __init__(self, threshold: float) -> None:
        """
        Initialize base bar aggregator.
        
        Args:
            threshold: Threshold value for bar completion
        """
        self.threshold = threshold
        
    @abstractmethod
    def calculate_cumulative_value(self, row: pd.Series, current_bar: Dict[str, Any]) -> float:
        """
        Calculate the value to add to cumulative total for threshold checking.
        
        Args:
            row: Current data row
            current_bar: Current bar being constructed
            
        Returns:
            Value to add to cumulative total
        """
        pass
        
    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """
        Get list of required columns for this aggregator.
        
        Returns:
            List of column names required in input DataFrame
        """
        pass
        
    def initialize_bar(self, row: pd.Series) -> Dict[str, Any]:
        """
        Initialize a new bar with data from first row.
        
        Args:
            row: First row of data for new bar
            
        Returns:
            Dictionary with initial bar values
        """
        return {
            'timestamp': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': 0.0,
            'trade_count': 0,
            'vwap': 0.0
        }
        
    def update_bar(self, current_bar: Dict[str, Any], row: pd.Series) -> None:
        """
        Update current bar with new row data.
        
        Args:
            current_bar: Bar to update
            row: New data row
        """
        current_bar['high'] = max(current_bar['high'], row['high'])
        current_bar['low'] = min(current_bar['low'], row['low'])
        current_bar['close'] = row['close']
        current_bar['volume'] += row.get('volume', 0)
        current_bar['trade_count'] += row.get('trade_count', 1)
        
    def finalize_bar(self, current_bar: Dict[str, Any], row: pd.Series) -> Dict[str, Any]:
        """
        Finalize bar before adding to results.
        
        Args:
            current_bar: Bar to finalize
            row: Last row of data for this bar
            
        Returns:
            Finalized bar dictionary
        """
        current_bar['timestamp_end'] = row['timestamp']
        return current_bar.copy()
        
    def add_calculated_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated columns to result DataFrame.
        
        Args:
            df: DataFrame with aggregated bars
            
        Returns:
            DataFrame with additional calculated columns
        """
        if not df.empty:
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            df['dv'] = df['hlc3'] * df['volume']
        return df
        
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert time-based bars to alternative bar type.
        
        Args:
            df: DataFrame with time-based OHLCV data
            
        Returns:
            DataFrame with aggregated bars
        """
        if df.empty:
            return pd.DataFrame()
            
        # Validate required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close'] + self.get_required_columns()
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        bars: list[Dict[str, Any]] = []
        current_bar: Dict[str, Any] = {}
        cumulative_value = 0.0
        
        for idx, row in df.iterrows():
            # Initialize new bar if needed
            if not current_bar:
                current_bar = self.initialize_bar(row)
                
            # Update current bar
            self.update_bar(current_bar, row)
            
            # Add to cumulative value
            cumulative_value += self.calculate_cumulative_value(row, current_bar)
            
            # Check if threshold reached
            if cumulative_value >= self.threshold:
                # Finalize and save bar
                bars.append(self.finalize_bar(current_bar, row))
                
                # Reset for next bar
                current_bar = {}
                cumulative_value = 0.0
                
        # Add final partial bar if exists
        if current_bar and cumulative_value > 0:
            bars.append(self.finalize_bar(current_bar, df.iloc[-1]))
            
        # Convert to DataFrame and add calculated columns
        result_df = pd.DataFrame(bars)
        return self.add_calculated_columns(result_df)


class DollarBarAggregator(BaseBarAggregator):
    """
    Generate dollar bars from time-based OHLCV data.
    
    Dollar bars sample data based on cumulative dollar value traded, creating bars
    when a threshold dollar amount has been reached. This provides more uniform
    information content per bar compared to time-based sampling.
    """
    
    def __init__(
        self, 
        threshold: float,
        price_column: str = "close",
        volume_column: str = "volume"
    ) -> None:
        """
        Initialize dollar bar aggregator.
        
        Args:
            threshold: Dollar value threshold for bar completion
            price_column: Column name for price data (default: "close")
            volume_column: Column name for volume data (default: "volume")
        """
        super().__init__(threshold)
        self.price_column = price_column
        self.volume_column = volume_column
        
    def get_required_columns(self) -> list[str]:
        """Get required columns for dollar bar aggregation."""
        return [self.price_column, self.volume_column]
        
    def calculate_cumulative_value(self, row: pd.Series, current_bar: Dict[str, Any]) -> float:
        """Calculate dollar value for the row."""
        return row[self.price_column] * row[self.volume_column]
    
    @staticmethod
    def estimate_threshold(
        df: pd.DataFrame,
        target_bars: int = 1000,
        price_column: str = "close",
        volume_column: str = "volume"
    ) -> float:
        """
        Estimate appropriate dollar threshold to achieve target number of bars.
        
        Args:
            df: Input DataFrame with OHLCV data
            target_bars: Desired number of dollar bars
            price_column: Column to use for price (default: "close")
            volume_column: Column to use for volume (default: "volume")
            
        Returns:
            Estimated dollar threshold
            
        Example:
            threshold = DollarBarAggregator.estimate_threshold(df, target_bars=500)
            aggregator = DollarBarAggregator(threshold)
            dollar_bars = aggregator.aggregate(df)
        """
        # Use dv column if available, otherwise calculate dollar volume
        if 'dv' in df.columns:
            total_dollar_volume = df['dv'].sum()
            print(f"Using existing dv column")
        else:
            dollar_values = df[price_column] * df[volume_column]
            total_dollar_volume = dollar_values.sum()
            print(f"Calculating dollar volume: {price_column} * {volume_column}")
        
        # Estimate threshold
        threshold = max(1, math.trunc(total_dollar_volume / target_bars))

        print(f"Estimated dollar threshold: ${threshold:,.2f} for ~{target_bars} bars")
        print(f"Total dollar volume: ${total_dollar_volume:,.2f}")
        
        return threshold


class VolumeBarAggregator(BaseBarAggregator):
    """
    Generate volume bars from time-based OHLCV data.
    
    Volume bars sample data based on cumulative volume traded, creating bars
    when a threshold volume amount has been reached. This provides bars that
    represent equal amounts of trading activity.
    """
    
    def __init__(
        self,
        threshold: float,
        volume_column: str = "volume"
    ) -> None:
        """
        Initialize volume bar aggregator.
        
        Args:
            threshold: Volume threshold for bar completion
            volume_column: Column name for volume data (default: "volume")
        """
        super().__init__(threshold)
        self.volume_column = volume_column
        self._volume_weighted_price_sum = 0.0
        
    def get_required_columns(self) -> list[str]:
        """Get required columns for volume bar aggregation."""
        return [self.volume_column]
        
    def calculate_cumulative_value(self, row: pd.Series, current_bar: Dict[str, Any]) -> float:
        """Calculate volume for the row."""
        # Track VWAP calculation
        typical_price = (row['high'] + row['low'] + row['close']) / 3
        self._volume_weighted_price_sum += typical_price * row[self.volume_column]
        return row[self.volume_column]
        
    def initialize_bar(self, row: pd.Series) -> Dict[str, Any]:
        """Initialize a new bar and reset VWAP tracking."""
        self._volume_weighted_price_sum = 0.0
        return super().initialize_bar(row)
        
    def finalize_bar(self, current_bar: Dict[str, Any], row: pd.Series) -> Dict[str, Any]:
        """Finalize bar with VWAP calculation."""
        bar = super().finalize_bar(current_bar, row)
        # Calculate VWAP
        if current_bar['volume'] > 0:
            bar['vwap'] = self._volume_weighted_price_sum / current_bar['volume']
        else:
            bar['vwap'] = current_bar['close']
        return bar
        
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to reset VWAP tracking between bars."""
        self._volume_weighted_price_sum = 0.0
        return super().aggregate(df)
    
    @staticmethod
    def estimate_threshold(
        df: pd.DataFrame,
        target_bars: int = 1000,
        volume_column: str = "volume",
        **kwargs
    ) -> float:
        """
        Estimate appropriate volume threshold to achieve target number of bars.
        
        Args:
            df: Input DataFrame with OHLCV data
            target_bars: Desired number of volume bars
            volume_column: Column to use for volume (default: "volume")
            **kwargs: Additional parameters (unused)
            
        Returns:
            Estimated volume threshold
            
        Example:
            threshold = VolumeBarAggregator.estimate_threshold(df, target_bars=500)
            aggregator = VolumeBarAggregator(threshold)
            volume_bars = aggregator.aggregate(df)
        """
        # Calculate total volume
        total_volume = df[volume_column].sum()
        
        # Estimate threshold
        threshold = total_volume / target_bars
        
        print(f"Estimated volume threshold: {threshold:,.2f} for ~{target_bars} bars")
        print(f"Total volume: {total_volume:,.2f}")
        
        return threshold


class TickBarAggregator(BaseBarAggregator):
    """
    Generate tick bars from time-based OHLCV data.
    
    Tick bars sample data based on number of trades, creating bars
    when a threshold number of trades has occurred. This provides bars
    that represent equal market activity regardless of volume or price.
    """
    
    def __init__(
        self,
        threshold: float,
        trade_count_column: str = "trade_count"
    ) -> None:
        """
        Initialize tick bar aggregator.
        
        Args:
            threshold: Number of trades threshold for bar completion
            trade_count_column: Column name for trade count data (default: "trade_count")
        """
        super().__init__(threshold)
        self.trade_count_column = trade_count_column
        
    def get_required_columns(self) -> list[str]:
        """Get required columns for tick bar aggregation."""
        return [self.trade_count_column]
        
    def calculate_cumulative_value(self, row: pd.Series, current_bar: Dict[str, Any]) -> float:
        """Calculate trade count for the row."""
        return row.get(self.trade_count_column, 1)  # Default to 1 if not present
    
    @staticmethod
    def estimate_threshold(
        df: pd.DataFrame,
        target_bars: int = 1000,
        trade_count_column: str = "trade_count",
        **kwargs
    ) -> float:
        """
        Estimate appropriate tick threshold to achieve target number of bars.
        
        Args:
            df: Input DataFrame with OHLCV data
            target_bars: Desired number of tick bars
            trade_count_column: Column with trade counts (default: "trade_count")
            **kwargs: Additional parameters (unused)
            
        Returns:
            Estimated tick threshold
        """
        # Calculate total trades
        if trade_count_column in df.columns:
            total_trades = df[trade_count_column].sum()
        else:
            # If no trade count column, assume 1 trade per row
            total_trades = len(df)
        
        # Estimate threshold
        threshold = total_trades / target_bars
        
        print(f"Estimated tick threshold: {threshold:,.0f} trades for ~{target_bars} bars")
        print(f"Total trades: {total_trades:,.0f}")
        
        return threshold
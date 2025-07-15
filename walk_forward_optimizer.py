"""
Walk-Forward Optimization (WFO) for backtesting trading strategies.

This module provides a WalkForwardOptimizer class that implements walk-forward
analysis using scikit-learn's TimeSeriesSplit. It divides historical data into
multiple train/test windows for robust strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from pydantic import BaseModel, Field


class SplitInfo(BaseModel):
    """Information about a single train/test split in walk-forward optimization."""
    
    split_id: int = Field(description="Sequential identifier for the split")
    train_start_idx: int = Field(description="Start index of training period", ge=0)
    train_end_idx: int = Field(description="End index of training period", ge=0)
    test_start_idx: int = Field(description="Start index of test period", ge=0)
    test_end_idx: int = Field(description="End index of test period", ge=0)
    train_size: int = Field(description="Number of samples in training set", gt=0)
    test_size: int = Field(description="Number of samples in test set", gt=0)
    train_indices: np.ndarray = Field(description="Array of training indices")
    test_indices: np.ndarray = Field(description="Array of test indices")
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True  # Allow numpy arrays
    
    @property
    def train_bars(self) -> int:
        """Number of bars in training period."""
        return self.train_size
    
    @property
    def test_bars(self) -> int:
        """Number of bars in test period."""
        return self.test_size
    
    @property
    def train_test_ratio(self) -> float:
        """Calculate ratio of training to test samples."""
        return self.train_size / self.test_size


class WalkForwardOptimizer:
    """
    Implements Walk-Forward Optimization for time series data.
    
    Walk-forward optimization is a backtesting methodology that divides data
    into multiple train/test periods, simulating how a strategy would perform
    in real-time with periodic re-optimization.
    
    Attributes:
        n_splits: Number of train/test splits (default: 5)
        test_size: Fixed size for test sets (optional)
        gap: Number of samples to exclude between train and test sets
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0
    ) -> None:
        """
        Initialize the Walk-Forward Optimizer.
        
        Args:
            n_splits: Number of splits for time series cross-validation
            test_size: Fixed size for each test set (if None, uses expanding window)
            gap: Number of samples to exclude between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap
        )
        
    def get_splits(self, df: pd.DataFrame) -> List[SplitInfo]:
        """
        Generate train/test splits for the given DataFrame.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            List of SplitInfo objects containing split information
        """        
        splits = []
        
        for i, (train_idx, test_idx) in enumerate(self.tscv.split(df)):
            split_info = SplitInfo(
                split_id=i + 1,
                train_start_idx=int(train_idx[0]),
                train_end_idx=int(train_idx[-1]),
                test_start_idx=int(test_idx[0]),
                test_end_idx=int(test_idx[-1]),
                train_size=len(train_idx),
                test_size=len(test_idx),
                train_indices=train_idx,
                test_indices=test_idx
            )
            splits.append(split_info)
            
        return splits
    
    def get_split_data(
        self, 
        df: pd.DataFrame, 
        split_info: SplitInfo
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract train and test DataFrames for a specific split.
        
        Args:
            df: Original DataFrame
            split_info: SplitInfo object from get_splits()
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = df.iloc[split_info.train_indices].copy()
        test_df = df.iloc[split_info.test_indices].copy()
        
        return train_df, test_df
    
    def print_splits_summary(self, splits: List[SplitInfo]) -> None:
        """
        Print a formatted summary of all splits.
        
        Args:
            splits: List of SplitInfo objects
        """
        print("\nWalk-Forward Optimization Splits Summary")
        print("=" * 80)
        
        for split in splits:
            print(f"\nSplit {split.split_id}:")
            print(f"  Training Indices: [{split.train_start_idx:,} to {split.train_end_idx:,}]")
            print(f"  Training Bars: {split.train_bars:,}")
            print(f"  Test Indices: [{split.test_start_idx:,} to {split.test_end_idx:,}]")
            print(f"  Test Bars: {split.test_bars:,}")
            print(f"  Train/Test Ratio: {split.train_test_ratio:.2f}")
            
            if self.gap > 0:
                print(f"  Gap Bars: {self.gap}")
        
        print("\n" + "=" * 80)
        
    def validate_data_sufficiency(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame has enough data for the requested number of splits.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is sufficient, False otherwise
        """
        min_samples = self.n_splits + 1
        if self.test_size:
            min_samples = self.n_splits + self.test_size
            
        if len(df) < min_samples:
            print(f"Warning: DataFrame has {len(df)} samples but needs at least {min_samples} for {self.n_splits} splits")
            return False
            
        return True
    
    def calculate_split_statistics(self, splits: List[SplitInfo]) -> dict[str, float]:
        """
        Calculate statistics about the splits.
        
        Args:
            splits: List of SplitInfo objects
            
        Returns:
            Dictionary with statistics:
            - avg_train_size: Average training set size
            - avg_test_size: Average test set size
            - total_train_bars: Total training bars across all splits
            - total_test_bars: Total test bars across all splits
            - train_test_ratio: Average ratio of train to test samples
        """
        train_sizes = [s.train_size for s in splits]
        test_sizes = [s.test_size for s in splits]
        
        train_bars = [s.train_bars for s in splits]
        test_bars = [s.test_bars for s in splits]
        
        stats = {
            'avg_train_size': np.mean(train_sizes),
            'avg_test_size': np.mean(test_sizes),
            'total_train_bars': sum(train_bars),
            'total_test_bars': sum(test_bars),
            'train_test_ratio': np.mean([t/s for t, s in zip(train_sizes, test_sizes)])
        }
        
        return stats
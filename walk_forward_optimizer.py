"""
Walk-Forward Optimization (WFO) for backtesting trading strategies.

This module provides a WalkForwardOptimizer class that implements walk-forward
analysis. It uses the SplitManager to divide historical data into
multiple train/test windows for robust strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
from split_manager import SplitManager, SplitInfo


class WalkForwardOptimizer:
    """
    Implements Walk-Forward Optimization for time series data.
    
    Walk-forward optimization is a backtesting methodology that divides data
    into multiple train/test periods, simulating how a strategy would perform
    in real-time with periodic re-optimization.
    
    Uses SplitManager for flexible split generation strategies.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_ratio: float = 0.3,  # Always use 70/30 split
        gap: int = 0
    ) -> None:
        """
        Initialize the Walk-Forward Optimizer.
        
        Args:
            n_splits: Number of splits for time series cross-validation
            test_ratio: Test size as fraction (0.3 = 30% test, 70% train)
            gap: Number of samples to exclude between train and test sets
        """
        self.split_manager = SplitManager(
            n_splits=n_splits,
            test_ratio=test_ratio,
            gap=gap
        )
        
    def get_splits(self, df: pd.DataFrame) -> List[SplitInfo]:
        """
        Generate train/test splits for the given DataFrame.
        
        Delegates to SplitManager for actual split generation.
        Uses hybrid sliding window approach by default.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            List of SplitInfo objects containing split information
        """
        return self.split_manager.generate_hybrid_splits(df)
    
    def get_split_data(
        self, 
        df: pd.DataFrame, 
        split_info: SplitInfo
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract train and test DataFrames for a specific split.
        
        Delegates to SplitManager.
        
        Args:
            df: Original DataFrame
            split_info: SplitInfo object from get_splits()
            
        Returns:
            Tuple of (train_df, test_df)
        """
        return self.split_manager.get_split_data(df, split_info)
    
    def print_splits_summary(self, splits: List[SplitInfo]) -> None:
        """
        Print a formatted summary of all splits.
        
        Delegates to SplitManager.
        
        Args:
            splits: List of SplitInfo objects
        """
        self.split_manager.print_splits_summary(splits)
        
    def validate_data_sufficiency(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame has enough data for the requested number of splits.
        
        Delegates to SplitManager.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is sufficient, False otherwise
        """
        return self.split_manager.validate_data_sufficiency(df)
    
    def calculate_split_statistics(self, splits: List[SplitInfo]) -> dict[str, float]:
        """
        Calculate statistics about the splits.
        
        Delegates to SplitManager.
        
        Args:
            splits: List of SplitInfo objects
            
        Returns:
            Dictionary with statistics
        """
        return self.split_manager.calculate_split_statistics(splits)
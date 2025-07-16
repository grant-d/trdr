"""
Split management for walk-forward optimization.

Handles generation and management of train/test splits for time series data.
Supports various splitting strategies including fixed ratio and hybrid sliding windows.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from pydantic import BaseModel, Field
import chalk


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


class SplitManager:
    """
    Manages generation of train/test splits for time series data.
    
    Supports multiple splitting strategies:
    - Fixed ratio splits (e.g., 70/30)
    - Hybrid sliding windows (overlapping train, non-overlapping test)
    - Expanding windows (not currently used but available)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_ratio: float = 0.3,
        gap: int = 0
    ) -> None:
        """
        Initialize the SplitManager.
        
        Args:
            n_splits: Number of splits to generate
            test_ratio: Test size as fraction (0.3 = 30% test, 70% train)
            gap: Number of samples to exclude between train and test sets
        """
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.gap = gap
    
    def generate_hybrid_splits(self, df: pd.DataFrame) -> List[SplitInfo]:
        """
        Generate train/test splits using hybrid sliding window approach.
        
        Training windows can overlap but test windows don't overlap.
        This is the recommended approach for regime-based strategies.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            List of SplitInfo objects containing split information
        """        
        total_length = len(df)
        
        # Calculate window sizes based on 70/30 split
        window_size = int(total_length * 0.7 / self.n_splits)  # Training window size
        test_size = int(window_size * self.test_ratio / (1 - self.test_ratio))  # 30% of total
        
        splits = []
        split_id = 1
        
        # Slide through the data with step size = test_size (non-overlapping test sets)
        for start in range(0, total_length - window_size - test_size + 1, test_size):
            # Training window
            train_start = start
            train_end = start + window_size - 1
            
            # Test window (with gap if specified)
            test_start = train_end + 1 + self.gap
            test_end = test_start + test_size - 1
            
            # Stop if test window would exceed data
            if test_end >= total_length:
                break
            
            # Create index arrays
            train_indices = np.arange(train_start, train_end + 1)
            test_indices = np.arange(test_start, test_end + 1)
            
            split_info = SplitInfo(
                split_id=split_id,
                train_start_idx=int(train_start),
                train_end_idx=int(train_end),
                test_start_idx=int(test_start),
                test_end_idx=int(test_end),
                train_size=len(train_indices),
                test_size=len(test_indices),
                train_indices=train_indices,
                test_indices=test_indices
            )
            splits.append(split_info)
            split_id += 1
            
            # Stop after n_splits
            if split_id > self.n_splits:
                break
            
        return splits
    
    def generate_fixed_splits(self, df: pd.DataFrame) -> List[SplitInfo]:
        """
        Generate non-overlapping train/test splits with fixed ratio.
        
        Each split uses independent data (no overlap between splits).
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            List of SplitInfo objects
        """
        total_length = len(df)
        split_size = total_length // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate boundaries for this split
            split_start = i * split_size
            split_end = (i + 1) * split_size if i < self.n_splits - 1 else total_length
            
            # Within the split, divide into train/test
            split_length = split_end - split_start
            train_size = int(split_length * (1 - self.test_ratio))
            
            train_start = split_start
            train_end = split_start + train_size - 1
            test_start = train_end + 1 + self.gap
            test_end = split_end - 1
            
            # Skip if not enough data for test
            if test_start >= test_end:
                continue
            
            train_indices = np.arange(train_start, train_end + 1)
            test_indices = np.arange(test_start, test_end + 1)
            
            split_info = SplitInfo(
                split_id=i + 1,
                train_start_idx=int(train_start),
                train_end_idx=int(train_end),
                test_start_idx=int(test_start),
                test_end_idx=int(test_end),
                train_size=len(train_indices),
                test_size=len(test_indices),
                train_indices=train_indices,
                test_indices=test_indices
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
            split_info: SplitInfo object from generate_splits()
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = df.iloc[split_info.train_indices].copy()
        test_df = df.iloc[split_info.test_indices].copy()
        
        return train_df, test_df
    
    def validate_data_sufficiency(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame has enough data for the requested number of splits.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is sufficient, False otherwise
        """
        min_samples = self.n_splits + 1
        
        if len(df) < min_samples:
            print(f"Warning: DataFrame has {len(df)} samples but needs at least {min_samples} for {self.n_splits} splits")
            return False
            
        return True
    
    def print_splits_summary(self, splits: List[SplitInfo]) -> None:
        """
        Print a formatted summary of all splits.
        
        Args:
            splits: List of SplitInfo objects
        """
        print(chalk.yellow + "\n" + "─" * 60 + chalk.RESET)
        print(chalk.yellow + chalk.bold + "📊 Walk-Forward Optimization Splits Summary" + chalk.RESET)
        print(chalk.yellow + "─" * 60 + chalk.RESET)
        
        for split in splits:
            print(chalk.blue + f"\nSplit {split.split_id}:" + chalk.RESET)
            print(f"  {chalk.white}Training:{chalk.RESET}  [{split.train_start_idx:>4} → {split.train_end_idx:<4}]  {chalk.bold}{split.train_size:>4}{chalk.RESET} bars")
            print(f"  {chalk.white}Testing:{chalk.RESET}   [{split.test_start_idx:>4} → {split.test_end_idx:<4}]  {chalk.bold}{split.test_size:>4}{chalk.RESET} bars")
            print(f"  {chalk.white}Ratio:{chalk.RESET}     {chalk.bold}{split.train_size/split.test_size:.2f}{chalk.RESET} : 1")
            
            if self.gap > 0:
                print(f"  {chalk.white}Gap:{chalk.RESET}       {chalk.bold}{self.gap}{chalk.RESET} bars")
        
        # Add summary statistics
        total_train = sum(s.train_size for s in splits)
        total_test = sum(s.test_size for s in splits)
        avg_ratio = sum(s.train_size/s.test_size for s in splits) / len(splits)
        
        print(chalk.yellow + "\nSummary:" + chalk.RESET)
        print(f"  {chalk.white}Total Training Bars:{chalk.RESET}  {chalk.bold}{total_train:>6}{chalk.RESET}")
        print(f"  {chalk.white}Total Test Bars:{chalk.RESET}      {chalk.bold}{total_test:>6}{chalk.RESET}")
        print(f"  {chalk.white}Average Ratio:{chalk.RESET}        {chalk.bold}{avg_ratio:.2f}{chalk.RESET} : 1")
        
        # Warning if insufficient data
        if any(s.train_size < 100 for s in splits):
            print(chalk.red + f"\n⚠️  Warning: Some splits have less than 100 training bars!" + chalk.RESET)
            print(chalk.red + f"   This may cause poor optimization results." + chalk.RESET)
        
        print(chalk.yellow + "─" * 60 + chalk.RESET)
    
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
"""
Type-safe argument classes for main.py functions
"""
from typing import Optional, Union


class AnalysisConfig:
    """Configuration for run_helios_analysis function"""
    
    def __init__(self,
                 use_dollar_bars: bool = False,
                 dollar_threshold: float = 1000000,
                 lookback: int = 20,
                 initial_capital: float = 100000,
                 max_position_pct: float = 0.95,
                 min_position_pct: float = 0.1,
                 save_results: bool = False,
                 output_dir: str = "./results"):
        self.use_dollar_bars = use_dollar_bars
        self.dollar_threshold = dollar_threshold
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.save_results = save_results
        self.output_dir = output_dir
    
    def get(self, key: str, default=None):
        """Dict-like interface for backward compatibility"""
        return getattr(self, key, default)


class OptimizeArgs:
    """Type-safe arguments for optimize command"""
    
    def __init__(self, 
                 data: str,
                 dollar_threshold: Optional[str] = None,
                 context_id: Optional[str] = None,
                 step_days: int = 90,
                 population: int = 50,
                 generations: int = 20,
                 fitness: str = "sortino",
                 allow_shorts: bool = False,
                 output: Optional[str] = None,
                 output_dir: str = "./optimization_results",
                 test: bool = False,
                 plot: bool = False,
                 llm: bool = False):
        self.data = data
        self.dollar_threshold = dollar_threshold
        self.context_id = context_id
        self.step_days = step_days  # Note: Not used in current implementation
        self.population = population
        self.generations = generations
        self.fitness = fitness
        self.allow_shorts = allow_shorts
        self.output = output
        self.output_dir = output_dir
        self.test = test
        self.plot = plot
        self.llm = llm


class TestArgs:
    """Type-safe arguments for test command"""
    
    def __init__(self,
                 params: str,
                 data: Optional[str] = None,
                 dollar_threshold: Optional[str] = None,
                 capital: float = 100000,
                 save_results: bool = False,
                 output_dir: str = "./strategy_results",
                 allow_shorts: bool = False,
                 plot: bool = False,
                 llm: bool = False):
        self.params = params
        self.data = data
        self.dollar_threshold = dollar_threshold
        self.capital = capital
        self.save_results = save_results
        self.output_dir = output_dir
        self.allow_shorts = allow_shorts
        self.plot = plot
        self.llm = llm
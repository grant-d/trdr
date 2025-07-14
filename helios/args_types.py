"""
Type-safe argument classes for main.py functions
"""
from typing import Optional


class OptimizeArgs:
    """Type-safe arguments for optimize command"""
    
    def __init__(self, 
                 data: str,
                 dollar_threshold: Optional[str] = None,
                 context_id: Optional[str] = None,
                 walk_forward: bool = False,
                 window_days: int = 365,
                 step_days: int = 90,
                 population: int = 50,
                 generations: int = 20,
                 fitness: str = "sortino",
                 allow_shorts: bool = False,
                 output: str = "optimization",
                 output_dir: str = "./optimization_results",
                 test: bool = False,
                 plot: bool = False,
                 llm: bool = False):
        self.data = data
        self.dollar_threshold = dollar_threshold
        self.context_id = context_id
        self.walk_forward = walk_forward
        self.window_days = window_days
        self.step_days = step_days
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
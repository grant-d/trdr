"""
Regime-specific playbooks for Helios Trader
Manages optimized parameters for different market regimes
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class RegimeParameters:
    """Parameters for a specific regime"""
    weights: Dict[str, float]
    lookback: int
    stop_loss_atr_multiplier: float
    position_size_multiplier: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class Playbook:
    """
    Manages regime-specific parameters for a trading context
    """
    
    # Default regime parameters (can be overridden)
    DEFAULT_REGIMES = {
        'Strong Bull': RegimeParameters(
            weights={'trend': 0.5, 'volatility': 0.2, 'exhaustion': 0.3},
            lookback=20,
            stop_loss_atr_multiplier=2.0,
            position_size_multiplier=1.0
        ),
        'Weak Bull': RegimeParameters(
            weights={'trend': 0.4, 'volatility': 0.3, 'exhaustion': 0.3},
            lookback=20,
            stop_loss_atr_multiplier=1.0,
            position_size_multiplier=0.7
        ),
        'Neutral': RegimeParameters(
            weights={'trend': 0.3, 'volatility': 0.4, 'exhaustion': 0.3},
            lookback=15,
            stop_loss_atr_multiplier=0.5,
            position_size_multiplier=0.0  # No position in neutral
        ),
        'Weak Bear': RegimeParameters(
            weights={'trend': 0.3, 'volatility': 0.3, 'exhaustion': 0.4},
            lookback=20,
            stop_loss_atr_multiplier=1.0,
            position_size_multiplier=0.7
        ),
        'Strong Bear': RegimeParameters(
            weights={'trend': 0.3, 'volatility': 0.2, 'exhaustion': 0.5},
            lookback=20,
            stop_loss_atr_multiplier=2.0,
            position_size_multiplier=1.0
        )
    }
    
    def __init__(self, context_id: str, playbook_dir: str = "./playbooks"):
        """
        Initialize a playbook for a specific context
        
        Parameters:
        -----------
        context_id : str
            Trading context ID
        playbook_dir : str
            Directory for playbook storage
        """
        self.context_id = context_id
        self.playbook_dir = Path(playbook_dir)
        self.playbook_dir.mkdir(exist_ok=True)
        self.playbook_file = self.playbook_dir / f"{context_id}_playbook.json"
        
        # Parse context for instrument/timeframe specific defaults
        self._parse_context_id()
        
        # Load or create playbook
        self.regimes = self._load_playbook()
    
    def _parse_context_id(self):
        """Parse context ID for instrument-specific settings"""
        parts = self.context_id.split('_')
        self.instrument = parts[0] if len(parts) > 0 else 'UNKNOWN'
        self.exchange = parts[1] if len(parts) > 1 else 'UNKNOWN'
        self.timeframe = parts[2] if len(parts) > 2 else '1h'
    
    def _load_playbook(self) -> Dict[str, RegimeParameters]:
        """Load playbook from file or create default"""
        if self.playbook_file.exists():
            with open(self.playbook_file, 'r') as f:
                data = json.load(f)
                return {
                    regime: RegimeParameters(**params)
                    for regime, params in data.items()
                }
        else:
            # Create instrument-specific defaults
            return self._create_default_playbook()
    
    def _create_default_playbook(self) -> Dict[str, RegimeParameters]:
        """Create instrument and timeframe specific default playbook"""
        playbook = {}
        
        # Adjust defaults based on instrument type
        if 'BTC' in self.instrument or 'ETH' in self.instrument:
            # Crypto: Higher volatility tolerance, different weights
            for regime, params in self.DEFAULT_REGIMES.items():
                adjusted = RegimeParameters(
                    weights=params.weights.copy(),
                    lookback=params.lookback,
                    stop_loss_atr_multiplier=params.stop_loss_atr_multiplier * 1.5,  # Wider stops
                    position_size_multiplier=params.position_size_multiplier * 0.8  # Smaller positions
                )
                # Increase volatility weight for crypto
                total_weight = sum(adjusted.weights.values())
                adjusted.weights['volatility'] *= 1.2
                # Renormalize
                new_total = sum(adjusted.weights.values())
                adjusted.weights = {k: v/new_total for k, v in adjusted.weights.items()}
                playbook[regime] = adjusted
        
        elif any(stock in self.instrument for stock in ['AAPL', 'GOOGL', 'MSFT', 'SPY']):
            # Stocks: More trend-following
            for regime, params in self.DEFAULT_REGIMES.items():
                adjusted = RegimeParameters(
                    weights=params.weights.copy(),
                    lookback=params.lookback,
                    stop_loss_atr_multiplier=params.stop_loss_atr_multiplier,
                    position_size_multiplier=params.position_size_multiplier
                )
                # Increase trend weight for stocks
                adjusted.weights['trend'] *= 1.2
                # Renormalize
                new_total = sum(adjusted.weights.values())
                adjusted.weights = {k: v/new_total for k, v in adjusted.weights.items()}
                playbook[regime] = adjusted
        
        else:
            # Default playbook
            playbook = self.DEFAULT_REGIMES.copy()
        
        # Adjust for timeframe
        if self.timeframe in ['1m', '5m', '15m']:
            # Short timeframes: Faster lookback, tighter stops
            for regime in playbook:
                playbook[regime].lookback = max(10, playbook[regime].lookback // 2)
                playbook[regime].stop_loss_atr_multiplier *= 0.7
        
        elif self.timeframe in ['1d', '1w']:
            # Long timeframes: Slower lookback, wider stops
            for regime in playbook:
                playbook[regime].lookback = min(50, playbook[regime].lookback * 2)
                playbook[regime].stop_loss_atr_multiplier *= 1.3
        
        return playbook
    
    def save_playbook(self):
        """Save playbook to file"""
        data = {
            regime: params.to_dict()
            for regime, params in self.regimes.items()
        }
        
        with open(self.playbook_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_parameters(self, regime: str) -> RegimeParameters:
        """Get parameters for a specific regime"""
        if regime not in self.regimes:
            # Return neutral parameters as fallback
            return self.regimes.get('Neutral', self.DEFAULT_REGIMES['Neutral'])
        return self.regimes[regime]
    
    def update_parameters(self, regime: str, parameters: Dict[str, Any]):
        """Update parameters for a specific regime"""
        if regime not in self.regimes:
            self.regimes[regime] = RegimeParameters(**parameters)
        else:
            # Update existing parameters
            current = self.regimes[regime]
            if 'weights' in parameters:
                current.weights.update(parameters['weights'])
            if 'lookback' in parameters:
                current.lookback = parameters['lookback']
            if 'stop_loss_atr_multiplier' in parameters:
                current.stop_loss_atr_multiplier = parameters['stop_loss_atr_multiplier']
            if 'position_size_multiplier' in parameters:
                current.position_size_multiplier = parameters['position_size_multiplier']
        
        self.save_playbook()
    
    def update_from_optimization(self, optimization_results: Dict[str, Dict[str, Any]]):
        """
        Update playbook from optimization results
        
        Parameters:
        -----------
        optimization_results : Dict[str, Dict[str, Any]]
            Results from genetic algorithm optimization
            Format: {regime: {weights: {...}, lookback: ..., ...}}
        """
        for regime, params in optimization_results.items():
            self.update_parameters(regime, params)
    
    def get_all_parameters(self) -> Dict[str, Dict]:
        """Get all regime parameters"""
        return {
            regime: params.to_dict()
            for regime, params in self.regimes.items()
        }
    
    def reset_to_defaults(self):
        """Reset playbook to default parameters"""
        self.regimes = self._create_default_playbook()
        self.save_playbook()


class PlaybookManager:
    """
    Manages playbooks for multiple trading contexts
    """
    
    def __init__(self, playbook_dir: str = "./playbooks"):
        """Initialize playbook manager"""
        self.playbook_dir = Path(playbook_dir)
        self.playbook_dir.mkdir(exist_ok=True)
        self.playbooks: Dict[str, Playbook] = {}
    
    def get_playbook(self, context_id: str) -> Playbook:
        """Get or create playbook for a context"""
        if context_id not in self.playbooks:
            self.playbooks[context_id] = Playbook(context_id, str(self.playbook_dir))
        return self.playbooks[context_id]
    
    def update_from_optimization(self, context_id: str, 
                               optimization_results: Dict[str, Dict[str, Any]]):
        """Update a context's playbook from optimization results"""
        playbook = self.get_playbook(context_id)
        playbook.update_from_optimization(optimization_results)
    
    def list_playbooks(self) -> list:
        """List all available playbooks"""
        playbook_files = self.playbook_dir.glob("*_playbook.json")
        return [f.stem.replace('_playbook', '') for f in playbook_files]
    
    def export_playbook(self, context_id: str, export_path: str):
        """Export a playbook to a file"""
        playbook = self.get_playbook(context_id)
        params = playbook.get_all_parameters()
        
        with open(export_path, 'w') as f:
            json.dump(params, f, indent=2)
    
    def import_playbook(self, context_id: str, import_path: str):
        """Import a playbook from a file"""
        with open(import_path, 'r') as f:
            params = json.load(f)
        
        playbook = self.get_playbook(context_id)
        for regime, regime_params in params.items():
            playbook.update_parameters(regime, regime_params)
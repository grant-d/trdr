"""
Trading Portfolio Tracker
Comprehensive system for tracking trades, positions, cash, and P&L with CSV persistence
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


class PortfolioTracker:
    """
    Comprehensive portfolio tracker with CSV persistence, slippage, and fee tracking.
    Tracks cash, positions, trades, and P&L for both crypto and equity trading.
    """
    
    def __init__(self, symbol: str, timeframe_minutes: int, initial_cash: float = 100000.0, 
                 data_dir: str = "./data/portfolio"):
        """
        Initialize portfolio tracker
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'BTCUSD', 'AAPL')
        timeframe_minutes : int
            Timeframe for the strategy
        initial_cash : float
            Starting cash balance
        data_dir : str
            Directory to store portfolio tracking files
        """
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.initial_cash = initial_cash
        
        # Setup file paths
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_")
        self.trades_filename = self.data_dir / f"{safe_symbol}_{timeframe_minutes}min_trades.csv"
        
        # Load existing trades or create new
        self.trades_df = self._load_trades()
        
        # Current state (calculated from trades)
        self.current_position_size = 0.0
        self.available_cash = initial_cash
        self.total_equity = initial_cash
        self._recalculate_position_from_trades()
        
    def _load_trades(self) -> pd.DataFrame:
        """Load existing trades from CSV or create new empty one"""
        if self.trades_filename.exists():
            try:
                df = pd.read_csv(self.trades_filename, parse_dates=['timestamp'])
                print(f"Loaded {len(df)} trade entries from {self.trades_filename.name}")
                return df
            except Exception as e:
                print(f"Error loading trades: {e}. Creating new file.")
                
        # Create new empty trades file with required columns
        columns = [
            'timestamp', 'symbol', 'side', 'quantity', 'price', 'market_price',
            'slippage_bps', 'fee_commission', 'fee_regulatory', 'fee_crypto', 
            'fee_total', 'trade_value', 'net_amount', 'position_before', 
            'position_after', 'cash_before', 'cash_after', 'realized_pnl',
            'notes'
        ]
        return pd.DataFrame(columns=columns)
    
    def _save_trades(self):
        """Save trades to CSV file"""
        try:
            self.trades_df.to_csv(self.trades_filename, index=False)
        except Exception as e:
            print(f"Error saving trades: {e}")
    
    def _recalculate_position_from_trades(self):
        """Recalculate current position and cash from trade history"""
        if self.trades_df.empty:
            self.current_position_size = 0.0
            self.available_cash = self.initial_cash
            self.total_equity = self.initial_cash
            return
            
        # Calculate position from all trades
        buy_qty = self.trades_df[self.trades_df['side'] == 'buy']['quantity'].sum()
        sell_qty = self.trades_df[self.trades_df['side'] == 'sell']['quantity'].sum()
        self.current_position_size = buy_qty - sell_qty
        
        # Get latest cash balance from trades
        if len(self.trades_df) > 0:
            self.available_cash = self.trades_df.iloc[-1]['cash_after']
        else:
            self.available_cash = self.initial_cash
            
        print(f"Position: {self.current_position_size:.6f} {self.symbol}, Cash: ${self.available_cash:.2f}")
    
    def calculate_alpaca_fees(self, trade_value: float, side: str, is_crypto: bool = False) -> Dict[str, float]:
        """Calculate Alpaca trading fees"""
        fees = {
            'commission': 0.0,  # No commission fees for US equities
            'regulatory': 0.0,  # Regulatory fees on sells
            'crypto': 0.0,      # Crypto trading fees
            'total': 0.0
        }
        
        if is_crypto:
            # 0.25% fee for crypto trades
            fees['crypto'] = trade_value * 0.0025
        elif side.lower() == 'sell':
            # Regulatory fees on sells (approximate)
            # SEC fee: $0.0000278 per $1 of sale proceeds
            # FINRA TAF: approximated as 0.01% of trade value
            fees['regulatory'] = trade_value * 0.0000278 + trade_value * 0.0001
        
        fees['total'] = fees['commission'] + fees['regulatory'] + fees['crypto']
        return fees
    
    def apply_slippage(self, price: float, quantity: float, side: str) -> Tuple[float, float]:
        """Apply realistic slippage to trade price
        
        Returns:
        --------
        Tuple[float, float]
            (execution_price, slippage_in_basis_points)
        """
        # Base slippage: 0.05% (5 basis points) for small trades
        base_slippage = 0.0005
        
        # Additional slippage based on trade size (market impact)
        trade_value = price * quantity
        if trade_value > 100000:  # Large trade
            size_slippage = 0.0015
        elif trade_value > 50000:  # Medium trade
            size_slippage = 0.001
        else:  # Small trade
            size_slippage = 0.0005
            
        total_slippage = base_slippage + size_slippage
        slippage_bps = total_slippage * 10000  # Convert to basis points
        
        if side.lower() == 'buy':
            # Buy higher due to slippage
            execution_price = price * (1 + total_slippage)
        else:
            # Sell lower due to slippage
            execution_price = price * (1 - total_slippage)
            
        return execution_price, slippage_bps
    
    def can_execute_trade(self, side: str, quantity: float, price: float) -> Tuple[bool, float, str]:
        """Check if trade can be executed and return adjusted quantity if needed
        
        Returns:
        --------
        Tuple[bool, float, str]
            (can_execute, adjusted_quantity, reason)
        """
        if side.lower() == 'buy':
            # Check if we have enough cash
            execution_price, _ = self.apply_slippage(price, quantity, side)
            trade_value = execution_price * quantity
            fees = self.calculate_alpaca_fees(trade_value, side, is_crypto='/' in self.symbol)
            total_cost = trade_value + fees['total']
            
            if total_cost <= self.available_cash:
                return True, quantity, "OK"
            else:
                # Calculate maximum quantity we can afford
                max_affordable = self.available_cash * 0.99  # Leave 1% buffer
                # Iteratively find max quantity accounting for slippage and fees
                test_qty = quantity * 0.5
                while test_qty > 0.001:  # Minimum quantity threshold
                    test_price, _ = self.apply_slippage(price, test_qty, side)
                    test_value = test_price * test_qty
                    test_fees = self.calculate_alpaca_fees(test_value, side, is_crypto='/' in self.symbol)
                    if test_value + test_fees['total'] <= max_affordable:
                        return True, test_qty, f"Reduced quantity due to insufficient cash"
                    test_qty *= 0.9
                
                return False, 0.0, f"Insufficient cash: need ${total_cost:.2f}, have ${self.available_cash:.2f}"
                
        else:  # sell
            # Check if we have enough position
            if quantity <= self.current_position_size:
                return True, quantity, "OK"
            else:
                if self.current_position_size > 0.001:  # Have some position
                    return True, self.current_position_size, f"Reduced quantity due to insufficient position"
                else:
                    return False, 0.0, f"Insufficient position: need {quantity:.6f}, have {self.current_position_size:.6f}"
    
    def record_trade(self, side: str, quantity: float, market_price: float, 
                    timestamp: Optional[datetime] = None, notes: str = "") -> Dict:
        """Record a trade in the ledger with slippage and fees
        
        Returns:
        --------
        Dict
            Trade details including execution price, fees, and P&L
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Check if trade can be executed
        can_execute, adjusted_qty, reason = self.can_execute_trade(side, quantity, market_price)
        if not can_execute:
            raise ValueError(f"Cannot execute trade: {reason}")
            
        if adjusted_qty != quantity:
            print(f"Trade quantity adjusted: {quantity:.6f} -> {adjusted_qty:.6f} ({reason})")
            quantity = adjusted_qty
        
        # Apply slippage
        execution_price, slippage_bps = self.apply_slippage(market_price, quantity, side)
        
        # Calculate trade value and fees
        trade_value = execution_price * quantity
        is_crypto = '/' in self.symbol
        fees = self.calculate_alpaca_fees(trade_value, side, is_crypto)
        
        # Calculate net amount (positive for cash in, negative for cash out)
        if side.lower() == 'buy':
            net_amount = -(trade_value + fees['total'])  # Cash out
        else:
            net_amount = trade_value - fees['total']  # Cash in
        
        # Calculate realized P&L for sells
        realized_pnl = 0.0
        if side.lower() == 'sell' and len(self.trades_df) > 0:
            # Simple FIFO P&L calculation (could be enhanced with proper cost basis tracking)
            buy_trades = self.trades_df[self.trades_df['side'] == 'buy']
            if not buy_trades.empty:
                avg_cost_basis = (buy_trades['trade_value'] + buy_trades['fee_total']).sum() / buy_trades['quantity'].sum()
                realized_pnl = (execution_price - avg_cost_basis) * quantity - fees['total']
        
        # Record positions and cash before trade
        position_before = self.current_position_size
        cash_before = self.available_cash
        
        # Update positions and cash
        if side.lower() == 'buy':
            self.current_position_size += quantity
        else:
            self.current_position_size -= quantity
            
        self.available_cash += net_amount
        
        # Create ledger entry
        entry = {
            'timestamp': timestamp,
            'symbol': self.symbol,
            'side': side.lower(),
            'quantity': quantity,
            'price': execution_price,
            'market_price': market_price,
            'slippage_bps': slippage_bps,
            'fee_commission': fees['commission'],
            'fee_regulatory': fees['regulatory'],
            'fee_crypto': fees['crypto'],
            'fee_total': fees['total'],
            'trade_value': trade_value,
            'net_amount': net_amount,
            'position_before': position_before,
            'position_after': self.current_position_size,
            'cash_before': cash_before,
            'cash_after': self.available_cash,
            'realized_pnl': realized_pnl,
            'notes': notes
        }
        
        # Add to trades DataFrame
        self.trades_df = pd.concat([self.trades_df, pd.DataFrame([entry])], ignore_index=True)
        
        # Save to CSV
        self._save_trades()
        
        # Print trade confirmation
        print(f"TRADE: {side.upper()} {quantity:.6f} {self.symbol} @ ${execution_price:.4f} "
              f"(slippage: {slippage_bps:.1f}bps, fees: ${fees['total']:.2f})")
        
        return entry
    
    def get_portfolio_value(self, current_price: float) -> Dict[str, float]:
        """Calculate current portfolio value
        
        Returns:
        --------
        Dict[str, float]
            Portfolio metrics including total value, unrealized P&L, etc.
        """
        position_value = self.current_position_size * current_price
        total_value = self.available_cash + position_value
        
        # Calculate total fees paid
        total_fees = self.trades_df['fee_total'].sum() if not self.trades_df.empty else 0.0
        
        # Calculate total realized P&L
        total_realized_pnl = self.trades_df['realized_pnl'].sum() if not self.trades_df.empty else 0.0
        
        # Calculate unrealized P&L
        if self.current_position_size > 0 and not self.trades_df.empty:
            buy_trades = self.trades_df[self.trades_df['side'] == 'buy']
            if not buy_trades.empty:
                total_cost = (buy_trades['trade_value'] + buy_trades['fee_total']).sum()
                avg_cost_basis = total_cost / buy_trades['quantity'].sum()
                unrealized_pnl = (current_price - avg_cost_basis) * self.current_position_size
            else:
                unrealized_pnl = 0.0
        else:
            unrealized_pnl = 0.0
        
        total_pnl = total_realized_pnl + unrealized_pnl
        
        return {
            'cash': self.available_cash,
            'position_size': self.current_position_size,
            'position_value': position_value,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'realized_pnl': total_realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_fees_paid': total_fees,
            'total_return_pct': (total_value - self.initial_cash) / self.initial_cash * 100
        }
    
    def get_trade_summary(self) -> Dict:
        """Get summary of all trading activity"""
        if self.trades_df.empty:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_volume': 0.0,
                'total_fees': 0.0,
                'avg_slippage_bps': 0.0
            }
        
        buy_trades = len(self.trades_df[self.trades_df['side'] == 'buy'])
        sell_trades = len(self.trades_df[self.trades_df['side'] == 'sell'])
        total_volume = self.trades_df['trade_value'].sum()
        total_fees = self.trades_df['fee_total'].sum()
        avg_slippage = self.trades_df['slippage_bps'].mean()
        
        return {
            'total_trades': len(self.trades_df),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_volume': total_volume,
            'total_fees': total_fees,
            'avg_slippage_bps': avg_slippage
        }
    
    def export_trades(self, filename: Optional[str] = None) -> str:
        """Export trades to CSV file
        
        Returns:
        --------
        str
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.symbol}_{self.timeframe_minutes}min_trades_{timestamp}.csv"
            
        export_path = self.data_dir / filename
        self.trades_df.to_csv(export_path, index=False)
        print(f"Trades exported to {export_path}")
        return str(export_path)
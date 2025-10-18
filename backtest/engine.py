"""
Backtesting Engine
Implements realistic backtesting with T+1 execution, slippage, fees, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Realistic backtesting engine with transaction costs and detailed performance tracking.
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 commission_pct: float = 0.001,
                 slippage_bps: float = 5,
                 execution_timing: str = 'close'):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission_pct: Commission as percentage (0.001 = 0.1%)
            slippage_bps: Slippage in basis points (5 = 0.05%)
            execution_timing: Price used for execution ('close', 'open', 'vwap')
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        self.execution_timing = execution_timing
        
        # State tracking
        self.cash = initial_capital
        self.positions = {}  # ticker -> shares
        self.portfolio_value = initial_capital
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.daily_stats = []
        self.positions_history = []
        
    def get_execution_price(self,
                           ticker: str,
                           date: pd.Timestamp,
                           market_data: pd.DataFrame,
                           is_buy: bool) -> Optional[float]:
        """
        Get execution price with slippage.
        
        Args:
            ticker: Stock ticker
            date: Execution date
            market_data: Market data DataFrame
            is_buy: True for buy, False for sell
            
        Returns:
            Execution price or None if unavailable
        """
        if date not in market_data.index:
            return None
        
        row = market_data.loc[date]
        
        # Base price based on timing
        if self.execution_timing == 'close':
            base_price = row['close']
        elif self.execution_timing == 'open':
            base_price = row['open']
        else:  # vwap or default to close
            base_price = row['close']
        
        # Apply slippage (adverse price movement)
        if is_buy:
            price = base_price * (1 + self.slippage_bps)
        else:
            price = base_price * (1 - self.slippage_bps)
        
        return price
    
    def calculate_transaction_cost(self,
                                  value: float) -> float:
        """
        Calculate transaction cost (commission).
        
        Args:
            value: Transaction value
            
        Returns:
            Commission cost
        """
        return value * self.commission_pct
    
    def execute_trades(self,
                      target_weights: Dict[str, float],
                      date: pd.Timestamp,
                      market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Execute trades to reach target portfolio weights.
        
        Args:
            target_weights: Target portfolio weights
            date: Execution date
            market_data: Dictionary of market data by ticker
            
        Returns:
            List of executed trades
        """
        trades = []
        
        # Calculate current weights efficiently
        current_value = self.portfolio_value
        current_weights = {}
        
        for ticker, shares in self.positions.items():
            if ticker not in market_data or date not in market_data[ticker].index:
                continue
            
            price = market_data[ticker].loc[date, 'close']
            position_value = shares * price
            current_weights[ticker] = position_value / current_value
        
        # Determine trades needed (combine all tickers at once)
        all_tickers = set(list(current_weights.keys()) + list(target_weights.keys()))
        
        # Minimum weight threshold for trading
        min_trade_threshold = 0.001
        
        for ticker in all_tickers:
            current_weight = current_weights.get(ticker, 0)
            target_weight = target_weights.get(ticker, 0)
            
            # Skip tiny changes
            if abs(target_weight - current_weight) < min_trade_threshold:
                continue
            
            if ticker not in market_data or date not in market_data[ticker].index:
                logger.warning(f"Cannot trade {ticker} on {date}: no data available")
                continue
            
            # Get execution price
            is_buy = target_weight > current_weight
            price = self.get_execution_price(ticker, date, market_data[ticker], is_buy)
            
            if price is None or price <= 0:
                continue
            
            # Calculate shares to trade
            target_value = target_weight * current_value
            current_shares = self.positions.get(ticker, 0)
            target_shares = int(target_value / price)
            shares_to_trade = target_shares - current_shares
            
            # Skip if no change needed
            if shares_to_trade == 0:
                continue
            
            # Execute trade
            trade_value = abs(shares_to_trade) * price
            commission = self.calculate_transaction_cost(trade_value)
            
            # Check if we have enough cash for buys
            if shares_to_trade > 0:  # Buy
                total_cost = trade_value + commission
                if total_cost > self.cash:
                    # Adjust to available cash (keep small buffer)
                    available_cash = self.cash * 0.99
                    shares_to_trade = int(available_cash / (price * (1 + self.commission_pct)))
                    if shares_to_trade <= 0:
                        continue
                    trade_value = shares_to_trade * price
                    commission = self.calculate_transaction_cost(trade_value)
                    total_cost = trade_value + commission
                
                self.cash -= total_cost
                self.positions[ticker] = current_shares + shares_to_trade
                
            else:  # Sell
                self.cash += trade_value - commission
                self.positions[ticker] = current_shares + shares_to_trade
                
                # Remove position if closed
                if self.positions[ticker] <= 0:
                    del self.positions[ticker]
            
            # Record trade
            trades.append({
                'date': date,
                'ticker': ticker,
                'shares': shares_to_trade,
                'price': price,
                'value': trade_value,
                'commission': commission,
                'side': 'buy' if shares_to_trade > 0 else 'sell'
            })
            
            logger.debug(f"Executed: {trades[-1]['side'].upper()} {abs(shares_to_trade)} {ticker} @ ${price:.2f}")
        
        self.trades.extend(trades)
        return trades
    
    def update_portfolio_value(self,
                              date: pd.Timestamp,
                              market_data: Dict[str, pd.DataFrame]):
        """
        Update portfolio value based on current prices.
        
        Args:
            date: Current date
            market_data: Market data dictionary
        """
        positions_value = 0
        
        for ticker, shares in self.positions.items():
            if ticker not in market_data or date not in market_data[ticker].index:
                # Use last known price
                continue
            
            price = market_data[ticker].loc[date, 'close']
            positions_value += shares * price
        
        self.portfolio_value = self.cash + positions_value
        
        # Record equity point
        self.equity_curve.append({
            'date': date,
            'equity': self.portfolio_value,
            'cash': self.cash,
            'positions_value': positions_value
        })
    
    def run_backtest(self,
                    strategy,
                    market_data: Dict[str, pd.DataFrame],
                    start_date: pd.Timestamp,
                    end_date: pd.Timestamp,
                    save_results: bool = True,
                    output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Run complete backtest for a strategy.
        
        Args:
            strategy: Strategy instance
            market_data: Dictionary of market data
            start_date: Backtest start date
            end_date: Backtest end date
            save_results: Save results to CSV
            output_dir: Output directory for results
            
        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Starting backtest: {start_date.date()} to {end_date.date()}")
        
        # Get rebalance dates
        rebalance_dates = strategy.get_rebalance_dates(start_date, end_date)
        
        # Track current holdings
        current_holdings = {}
        
        for rebal_date in rebalance_dates:
            if rebal_date > end_date:
                break
            
            logger.info(f"Rebalancing on {rebal_date.date()}...")
            
            # Generate signals (T+0)
            signals = strategy.generate_signals(market_data, rebal_date)
            
            if signals.empty:
                logger.warning(f"No signals generated on {rebal_date.date()}")
                continue
            
            # Select portfolio
            target_portfolio = strategy.select_portfolio(signals, current_holdings, rebal_date)
            
            # Apply execution delay (T+1)
            execution_date = strategy.apply_execution_delay(rebal_date)
            
            # Execute trades
            trades = self.execute_trades(target_portfolio, execution_date, market_data)
            
            # Update current holdings
            current_holdings = target_portfolio
            
            # Update portfolio value
            self.update_portfolio_value(execution_date, market_data)
            
            # Record positions
            self.positions_history.append({
                'date': execution_date,
                'positions': self.positions.copy(),
                'weights': target_portfolio.copy(),
                'n_positions': len(self.positions)
            })
        
        # Generate final results
        results = self.generate_results(save_results, output_dir)
        
        logger.info(f"Backtest complete. Final equity: ${self.portfolio_value:,.2f}")
        
        return results
    
    def generate_results(self,
                        save: bool = True,
                        output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Generate backtest results and performance metrics.
        
        Args:
            save: Save results to CSV
            output_dir: Output directory
            
        Returns:
            Results DataFrame
        """
        from pathlib import Path
        
        # Equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Calculate drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        
        # Performance metrics
        total_return = (self.portfolio_value / self.initial_capital) - 1
        returns = equity_df['returns'].dropna()
        
        # Annualize metrics (assuming daily data)
        trading_days = len(equity_df)
        years = trading_days / 252
        
        metrics = {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (1 / years) - 1 if years > 0 else 0,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'max_drawdown': equity_df['drawdown'].min(),
            'total_trades': len(self.trades),
            'final_value': self.portfolio_value,
            'initial_value': self.initial_capital
        }
        
        logger.info("\n=== Backtest Performance ===")
        for key, value in metrics.items():
            if 'return' in key or 'drawdown' in key or 'volatility' in key or 'sharpe' in key:
                logger.info(f"{key}: {value:.2%}" if value < 10 else f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Save results if requested
        if save and output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save equity curve
            equity_df.to_csv(out_path / f'equity_curve_{timestamp}.csv', index=False)
            
            # Save trades
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(out_path / f'trades_{timestamp}.csv', index=False)
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(out_path / f'metrics_{timestamp}.csv', index=False)
            
            logger.info(f"Results saved to {output_dir}")
        
        return equity_df


if __name__ == "__main__":
    print("Backtest Engine initialized")

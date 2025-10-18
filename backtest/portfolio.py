"""
Portfolio Construction Module
Implements risk-adjusted portfolio weighting, filters, and position selection logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """
    Constructs portfolios with risk management, filters, and optimization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize portfolio constructor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Portfolio parameters
        self.max_positions = self.config.get('max_positions', 20)
        self.min_positions = self.config.get('min_positions', 5)
        self.max_position_weight = self.config.get('max_position_weight', 0.10)
        
        # Risk parameters
        self.target_volatility = self.config.get('target_volatility', 0.15)
        self.max_turnover = self.config.get('max_turnover', 0.50)
        
        # Filters
        self.min_liquidity_rank = self.config.get('min_liquidity_rank', 0.5)
        self.max_volatility = self.config.get('max_volatility', 0.50)
        
    def apply_filters(self,
                     signals: pd.DataFrame,
                     market_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Apply filters to candidate stocks.
        
        Args:
            signals: DataFrame with trading signals
            market_data: Optional market data for additional filtering
            
        Returns:
            Filtered signals DataFrame
        """
        filtered = signals.copy()
        
        # Filter by volatility if available
        if 'volatility' in filtered.columns:
            filtered = filtered[filtered['volatility'] <= self.max_volatility]
            logger.debug(f"After volatility filter: {len(filtered)} stocks")
        
        # Filter by liquidity if available
        if 'liquidity_rank' in filtered.columns:
            filtered = filtered[filtered['liquidity_rank'] >= self.min_liquidity_rank]
            logger.debug(f"After liquidity filter: {len(filtered)} stocks")
        
        # Remove stocks with missing key data
        filtered = filtered.dropna(subset=['predicted_return'])
        
        return filtered
    
    def rank_stocks(self,
                   signals: pd.DataFrame,
                   ranking_method: str = 'risk_adjusted') -> pd.DataFrame:
        """
        Rank stocks for portfolio selection.
        
        Args:
            signals: Signals DataFrame
            ranking_method: Method for ranking ('simple', 'risk_adjusted', 'confidence_weighted')
            
        Returns:
            Ranked signals DataFrame with 'rank' column
        """
        ranked = signals.copy()
        
        if ranking_method == 'simple':
            # Simple ranking by predicted return
            ranked['score'] = ranked['predicted_return']
            
        elif ranking_method == 'risk_adjusted':
            # Risk-adjusted score (return / volatility)
            if 'volatility' in ranked.columns:
                # Avoid division by zero
                ranked['score'] = ranked['predicted_return'] / ranked['volatility'].replace(0, np.nan)
            else:
                ranked['score'] = ranked['predicted_return']
                
        elif ranking_method == 'confidence_weighted':
            # Confidence-weighted return
            if 'confidence' in ranked.columns:
                ranked['score'] = ranked['predicted_return'] * ranked['confidence']
            else:
                ranked['score'] = ranked['predicted_return']
        
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        
        # Sort by score (descending) and assign ranks
        ranked.sort_values('score', ascending=False, inplace=True)
        ranked['rank'] = range(1, len(ranked) + 1)
        
        return ranked
    
    def select_top_stocks(self,
                         ranked_signals: pd.DataFrame,
                         n_stocks: Optional[int] = None) -> pd.DataFrame:
        """
        Select top N stocks from ranked signals.
        
        Args:
            ranked_signals: Ranked signals DataFrame
            n_stocks: Number of stocks to select (None = use max_positions)
            
        Returns:
            Selected stocks DataFrame
        """
        if n_stocks is None:
            n_stocks = self.max_positions
        
        selected = ranked_signals.head(n_stocks)
        
        if len(selected) < self.min_positions:
            logger.warning(f"Only {len(selected)} stocks selected (min={self.min_positions})")
        
        return selected
    
    def calculate_weights(self,
                         selected_stocks: pd.DataFrame,
                         weighting_method: str = 'risk_parity') -> pd.DataFrame:
        """
        Calculate portfolio weights for selected stocks.
        
        Args:
            selected_stocks: Selected stocks DataFrame
            weighting_method: Weighting method ('equal', 'risk_parity', 'inverse_vol', 'prediction')
            
        Returns:
            DataFrame with 'weight' column added
        """
        stocks = selected_stocks.copy()
        n_stocks = len(stocks)
        
        if n_stocks == 0:
            return stocks
        
        if weighting_method == 'equal':
            # Equal weighting
            stocks['weight'] = 1.0 / n_stocks
            
        elif weighting_method in ('risk_parity', 'inverse_vol'):
            # Inverse volatility weighting
            if 'volatility' not in stocks.columns:
                logger.warning("Volatility not available, using equal weights")
                stocks['weight'] = 1.0 / n_stocks
            else:
                # Avoid division by zero
                volatilities = stocks['volatility'].replace(0, np.nan)
                stocks['inv_vol'] = 1.0 / volatilities
                stocks['weight'] = stocks['inv_vol'] / stocks['inv_vol'].sum()
                
        elif weighting_method == 'prediction':
            # Weight by predicted return magnitude
            stocks['abs_pred'] = stocks['predicted_return'].abs()
            stocks['weight'] = stocks['abs_pred'] / stocks['abs_pred'].sum()
            
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")
        
        # Apply max weight constraint and normalize
        stocks['weight'] = stocks['weight'].clip(upper=self.max_position_weight)
        total_weight = stocks['weight'].sum()
        if total_weight > 0:
            stocks['weight'] = stocks['weight'] / total_weight
        
        return stocks
    
    def apply_turnover_constraint(self,
                                 target_weights: Dict[str, float],
                                 current_weights: Dict[str, float],
                                 max_turnover: Optional[float] = None) -> Dict[str, float]:
        """
        Apply turnover constraint to limit portfolio changes.
        
        Args:
            target_weights: Target portfolio weights
            current_weights: Current portfolio weights
            max_turnover: Maximum allowed turnover (None = use config)
            
        Returns:
            Adjusted target weights
        """
        if max_turnover is None:
            max_turnover = self.max_turnover
        
        # Calculate current turnover
        all_tickers = set(list(target_weights.keys()) + list(current_weights.keys()))
        turnover = sum(
            abs(target_weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        )
        
        # If under limit, no adjustment needed
        if turnover <= max_turnover:
            return target_weights
        
        # Scale down changes proportionally
        scale_factor = max_turnover / turnover
        
        adjusted_weights = {}
        for ticker in all_tickers:
            current_w = current_weights.get(ticker, 0)
            target_w = target_weights.get(ticker, 0)
            
            # Scale the change
            change = (target_w - current_w) * scale_factor
            adjusted_w = current_w + change
            
            if adjusted_w > 0.001:  # Minimum weight threshold
                adjusted_weights[ticker] = adjusted_w
        
        # Normalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {t: w/total for t, w in adjusted_weights.items()}
        
        logger.info(f"Applied turnover constraint: {turnover:.2%} -> {max_turnover:.2%}")
        
        return adjusted_weights
    
    def construct_portfolio(self,
                           signals: pd.DataFrame,
                           current_holdings: Dict[str, float],
                           market_data: Optional[Dict[str, pd.DataFrame]] = None,
                           ranking_method: str = 'risk_adjusted',
                           weighting_method: str = 'risk_parity') -> Dict[str, float]:
        """
        Complete portfolio construction pipeline.
        
        Args:
            signals: Trading signals DataFrame
            current_holdings: Current portfolio holdings
            market_data: Optional market data
            ranking_method: Method for ranking stocks
            weighting_method: Method for calculating weights
            
        Returns:
            Target portfolio weights dictionary
        """
        # Apply filters
        filtered = self.apply_filters(signals, market_data)
        
        if filtered.empty:
            logger.warning("No stocks passed filters")
            return {}
        
        # Rank stocks
        ranked = self.rank_stocks(filtered, ranking_method)
        
        # Select top stocks
        selected = self.select_top_stocks(ranked)
        
        if selected.empty:
            logger.warning("No stocks selected")
            return {}
        
        # Calculate weights
        weighted = self.calculate_weights(selected, weighting_method)
        
        # Create target portfolio
        target_portfolio = dict(zip(weighted['ticker'], weighted['weight']))
        
        # Apply turnover constraint
        final_portfolio = self.apply_turnover_constraint(
            target_portfolio,
            current_holdings
        )
        
        logger.info(f"Constructed portfolio with {len(final_portfolio)} positions")
        
        return final_portfolio
    
    def calculate_portfolio_stats(self,
                                 weights: Dict[str, float],
                                 returns: Optional[Dict[str, pd.Series]] = None,
                                 volatilities: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate portfolio-level statistics.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns for each stock
            volatilities: Volatility estimates for each stock
            
        Returns:
            Dictionary of portfolio statistics
        """
        stats = {
            'n_positions': len(weights),
            'max_weight': max(weights.values()) if weights else 0,
            'min_weight': min(weights.values()) if weights else 0,
            'concentration': sum([w**2 for w in weights.values()])  # Herfindahl index
        }
        
        # Expected return (if available)
        if returns is not None:
            expected_return = sum(
                weights.get(ticker, 0) * returns[ticker].mean()
                for ticker in weights.keys()
                if ticker in returns
            )
            stats['expected_return'] = expected_return
        
        # Portfolio volatility (if available)
        if volatilities is not None:
            # Simplified: assume zero correlation
            portfolio_vol = np.sqrt(sum(
                (weights.get(ticker, 0) ** 2) * (volatilities.get(ticker, 0) ** 2)
                for ticker in weights.keys()
                if ticker in volatilities
            ))
            stats['volatility'] = portfolio_vol
        
        return stats


if __name__ == "__main__":
    # Example usage
    constructor = PortfolioConstructor({
        'max_positions': 20,
        'max_position_weight': 0.10,
        'max_turnover': 0.50
    })
    
    # Sample signals
    signals_data = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'signal': [1, 1, 1, 1, 1],
        'predicted_return': [0.05, 0.08, 0.06, 0.07, 0.10],
        'volatility': [0.20, 0.18, 0.22, 0.25, 0.35],
        'confidence': [0.8, 0.9, 0.85, 0.75, 0.7]
    })
    
    # Construct portfolio
    portfolio = constructor.construct_portfolio(
        signals_data,
        current_holdings={},
        ranking_method='risk_adjusted',
        weighting_method='risk_parity'
    )
    
    print("Constructed Portfolio:")
    for ticker, weight in sorted(portfolio.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ticker}: {weight:.2%}")

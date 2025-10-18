"""
Base Strategy Module
Provides abstract base class for trading strategies with plug-and-play architecture.
Supports XML and Python template definitions.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    All strategies should inherit from this class and implement required methods.
    """
    
    def __init__(self, 
                 name: str,
                 universe: List[str],
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            universe: List of ticker symbols in universe
            config: Strategy configuration parameters
        """
        self.name = name
        self.universe = universe
        self.config = config or {}
        
        # Strategy parameters
        self.max_positions = self.config.get('max_positions', 20)
        self.min_positions = self.config.get('min_positions', 5)
        self.rebalance_frequency = self.config.get('rebalance_frequency', 'monthly')
        self.execution_delay = self.config.get('execution_delay', 1)  # T+1
        
        # State tracking
        self.current_positions = {}
        self.signals = pd.DataFrame()
        self.predictions = pd.DataFrame()
        
    @abstractmethod
    def generate_signals(self, 
                        data: Dict[str, pd.DataFrame],
                        date: pd.Timestamp) -> pd.DataFrame:
        """
        Generate trading signals for all stocks at given date.
        
        Args:
            data: Dictionary mapping ticker to DataFrame with features
            date: Current date for signal generation
            
        Returns:
            DataFrame with columns: ['ticker', 'signal', 'confidence', 'predicted_return']
            signal: -1 (sell), 0 (hold), 1 (buy)
        """
        pass
    
    @abstractmethod
    def select_portfolio(self,
                        signals: pd.DataFrame,
                        current_holdings: Dict[str, float],
                        date: pd.Timestamp) -> Dict[str, float]:
        """
        Select portfolio based on signals and current holdings.
        
        Args:
            signals: DataFrame with trading signals
            current_holdings: Current portfolio holdings (ticker -> weight)
            date: Current date
            
        Returns:
            Dictionary mapping ticker to target weight
        """
        pass
    
    def get_rebalance_dates(self,
                           start_date: pd.Timestamp,
                           end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """
        Get list of rebalance dates based on frequency.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of rebalance dates
        """
        if self.rebalance_frequency == 'daily':
            dates = pd.bdate_range(start_date, end_date, freq='D')
        elif self.rebalance_frequency == 'weekly':
            dates = pd.bdate_range(start_date, end_date, freq='W-FRI')
        elif self.rebalance_frequency == 'monthly':
            dates = pd.bdate_range(start_date, end_date, freq='BM')
        elif self.rebalance_frequency == 'quarterly':
            dates = pd.bdate_range(start_date, end_date, freq='BQ')
        else:
            raise ValueError(f"Invalid rebalance frequency: {self.rebalance_frequency}")
        
        return list(dates)
    
    def apply_execution_delay(self, signal_date: pd.Timestamp) -> pd.Timestamp:
        """
        Apply execution delay (T+N logic).
        
        Args:
            signal_date: Date when signal was generated
            
        Returns:
            Date when trade should be executed
        """
        # Add execution delay (business days)
        execution_date = signal_date + pd.Timedelta(days=self.execution_delay)
        
        # Ensure it's a business day
        while execution_date.weekday() >= 5:  # Saturday=5, Sunday=6
            execution_date += pd.Timedelta(days=1)
        
        return execution_date
    
    def calculate_turnover(self,
                          current_weights: Dict[str, float],
                          target_weights: Dict[str, float]) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            Turnover as fraction (0 to 2, where 2 = complete portfolio flip)
        """
        all_tickers = set(list(current_weights.keys()) + list(target_weights.keys()))
        
        turnover = 0.0
        for ticker in all_tickers:
            current_weight = current_weights.get(ticker, 0.0)
            target_weight = target_weights.get(ticker, 0.0)
            turnover += abs(target_weight - current_weight)
        
        return turnover
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', universe_size={len(self.universe)})"


class MLStrategy(BaseStrategy):
    """
    Machine learning based strategy.
    Generates signals from ML model predictions with risk-adjusted portfolio construction.
    """
    
    def __init__(self,
                 name: str,
                 universe: List[str],
                 model_predictions: Dict[str, pd.DataFrame],
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML strategy.
        
        Args:
            name: Strategy name
            universe: List of tickers
            model_predictions: Dictionary mapping ticker to predictions DataFrame
            config: Strategy configuration
        """
        super().__init__(name, universe, config)
        self.model_predictions = model_predictions
        
        # ML-specific parameters
        self.prediction_col = self.config.get('prediction_col', 'predicted_return')
        self.confidence_col = self.config.get('confidence_col', 'confidence')
        self.volatility_col = self.config.get('volatility_col', 'volatility')
        
        # Portfolio construction parameters
        self.use_risk_weighting = self.config.get('use_risk_weighting', True)
        self.use_confidence_weighting = self.config.get('use_confidence_weighting', True)
        self.max_position_weight = self.config.get('max_position_weight', 0.10)
        
    def generate_signals(self,
                        data: Dict[str, pd.DataFrame],
                        date: pd.Timestamp) -> pd.DataFrame:
        """
        Generate signals from ML predictions.
        
        Args:
            data: Dictionary with ticker data (not used directly, uses model_predictions)
            date: Current date
            
        Returns:
            DataFrame with signals
        """
        signals_list = []
        
        for ticker in self.universe:
            if ticker not in self.model_predictions:
                continue
            
            pred_df = self.model_predictions[ticker]
            
            # Get prediction for current date
            if date not in pred_df.index:
                continue
            
            pred_row = pred_df.loc[date]
            
            # Extract prediction and confidence
            predicted_return = pred_row.get(self.prediction_col, np.nan)
            confidence = pred_row.get(self.confidence_col, 1.0)
            volatility = pred_row.get(self.volatility_col, pred_row.get('volatility_20', np.nan))
            
            if np.isnan(predicted_return):
                continue
            
            # Generate signal based on predicted return
            if predicted_return > 0.01:  # 1% threshold
                signal = 1  # Buy
            elif predicted_return < -0.01:
                signal = -1  # Sell
            else:
                signal = 0  # Hold
            
            signals_list.append({
                'ticker': ticker,
                'signal': signal,
                'predicted_return': predicted_return,
                'confidence': confidence,
                'volatility': volatility
            })
        
        if not signals_list:
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(signals_list)
        return signals_df
    
    def select_portfolio(self,
                        signals: pd.DataFrame,
                        current_holdings: Dict[str, float],
                        date: pd.Timestamp) -> Dict[str, float]:
        """
        Select portfolio using risk-adjusted expected returns.
        
        Args:
            signals: DataFrame with signals
            current_holdings: Current holdings
            date: Current date
            
        Returns:
            Target portfolio weights
        """
        if signals.empty:
            return {}
        
        # Filter for buy signals
        buy_signals = signals[signals['signal'] == 1].copy()
        
        if buy_signals.empty:
            # No buy signals, liquidate everything
            return {}
        
        # Calculate scores for ranking
        buy_signals['score'] = buy_signals['predicted_return']
        
        # Apply confidence weighting
        if self.use_confidence_weighting and 'confidence' in buy_signals.columns:
            buy_signals['score'] *= buy_signals['confidence']
        
        # Apply inverse volatility weighting (risk-adjusted)
        if self.use_risk_weighting and 'volatility' in buy_signals.columns:
            buy_signals = buy_signals[buy_signals['volatility'].notna()]
            if not buy_signals.empty:
                # Inverse volatility (higher vol = lower weight)
                buy_signals['inv_vol'] = 1.0 / buy_signals['volatility']
                buy_signals['score'] *= buy_signals['inv_vol']
        
        # Rank stocks by score
        buy_signals = buy_signals.sort_values('score', ascending=False)
        
        # Select top N stocks
        selected = buy_signals.head(self.max_positions)
        
        # Check minimum positions
        if len(selected) < self.min_positions:
            logger.warning(f"Only {len(selected)} positions available (min={self.min_positions})")
            if len(selected) == 0:
                return {}
        
        # Calculate weights
        if self.use_risk_weighting and 'volatility' in selected.columns:
            # Risk-parity-like weighting (inverse volatility)
            selected['weight'] = selected['inv_vol'] / selected['inv_vol'].sum()
        else:
            # Equal weighting
            selected['weight'] = 1.0 / len(selected)
        
        # Apply max position constraint
        selected['weight'] = selected['weight'].clip(upper=self.max_position_weight)
        
        # Normalize weights to sum to 1
        total_weight = selected['weight'].sum()
        if total_weight > 0:
            selected['weight'] = selected['weight'] / total_weight
        
        # Create target portfolio dictionary
        target_portfolio = dict(zip(selected['ticker'], selected['weight']))
        
        return target_portfolio


class StrategyLoader:
    """
    Loads strategies from XML templates or Python files.
    Enables plug-and-play strategy development.
    """
    
    @staticmethod
    def load_from_xml(xml_path: str) -> Dict[str, Any]:
        """
        Load strategy configuration from XML file.
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            Dictionary with strategy configuration
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        config = {
            'name': root.find('name').text,
            'type': root.find('type').text,
            'parameters': {}
        }
        
        # Parse parameters
        params = root.find('parameters')
        if params is not None:
            for param in params:
                param_name = param.tag
                param_value = param.text
                
                # Try to convert to appropriate type
                try:
                    if '.' in param_value:
                        param_value = float(param_value)
                    else:
                        param_value = int(param_value)
                except ValueError:
                    # Keep as string
                    pass
                
                config['parameters'][param_name] = param_value
        
        return config
    
    @staticmethod
    def create_xml_template(filepath: str, 
                           name: str,
                           strategy_type: str = 'ml',
                           parameters: Optional[Dict[str, Any]] = None):
        """
        Create XML template file for a strategy.
        
        Args:
            filepath: Path to save XML file
            name: Strategy name
            strategy_type: Type of strategy
            parameters: Strategy parameters
        """
        root = ET.Element('strategy')
        
        # Add basic info
        ET.SubElement(root, 'name').text = name
        ET.SubElement(root, 'type').text = strategy_type
        
        # Add parameters
        params_elem = ET.SubElement(root, 'parameters')
        if parameters:
            for key, value in parameters.items():
                ET.SubElement(params_elem, key).text = str(value)
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Created strategy template: {filepath}")


if __name__ == "__main__":
    # Example: Create XML template
    StrategyLoader.create_xml_template(
        'strategies/templates/ml_strategy_template.xml',
        name='ML Mean Reversion Strategy',
        strategy_type='ml',
        parameters={
            'max_positions': 20,
            'rebalance_frequency': 'monthly',
            'use_risk_weighting': True,
            'max_position_weight': 0.10
        }
    )
    
    print("Strategy template created successfully")

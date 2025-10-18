"""
Technical Indicators Feature Module
Implements common technical indicators as plug-and-play features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from features.base_feature import BaseFeature, FeatureRegistry
import logging

logger = logging.getLogger(__name__)


@FeatureRegistry.register('returns')
class ReturnsFeature(BaseFeature):
    """Calculate various return metrics."""
    
    def __init__(self, name: str = 'returns', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.periods = config.get('periods', [1, 5, 21]) if config else [1, 5, 21]
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute return features (vectorized)."""
        features = pd.DataFrame(index=data.index)
        
        # Vectorized computation for all periods
        close_prices = data['close']
        log_close = np.log(close_prices)
        
        for period in self.periods:
            # Simple returns
            features[f'return_{period}d'] = close_prices.pct_change(period)
            
            # Log returns (more efficient using pre-computed log)
            features[f'log_return_{period}d'] = log_close - log_close.shift(period)
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('sma')
class SMAFeature(BaseFeature):
    """Simple Moving Average indicator."""
    
    def __init__(self, name: str = 'sma', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.periods = config.get('periods', [20, 50, 200]) if config else [20, 50, 200]
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute SMA features (vectorized)."""
        features = pd.DataFrame(index=data.index)
        close_prices = data['close']
        
        for period in self.periods:
            sma = close_prices.rolling(window=period, min_periods=period).mean()
            features[f'sma_{period}'] = sma
            features[f'sma_{period}_ratio'] = close_prices / sma
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('ema')
class EMAFeature(BaseFeature):
    """Exponential Moving Average indicator."""
    
    def __init__(self, name: str = 'ema', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.periods = config.get('periods', [12, 26]) if config else [12, 26]
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute EMA features (vectorized)."""
        features = pd.DataFrame(index=data.index)
        close_prices = data['close']
        
        for period in self.periods:
            ema = close_prices.ewm(span=period, adjust=False, min_periods=period).mean()
            features[f'ema_{period}'] = ema
            features[f'ema_{period}_ratio'] = close_prices / ema
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('rsi')
class RSIFeature(BaseFeature):
    """Relative Strength Index indicator."""
    
    def __init__(self, name: str = 'rsi', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.period = config.get('period', 14) if config else 14
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI (vectorized)."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses (vectorized)
        gain = delta.clip(lower=0).rolling(window=self.period, min_periods=self.period).mean()
        loss = (-delta).clip(lower=0).rolling(window=self.period, min_periods=self.period).mean()
        
        # Calculate RS and RSI (avoid division by zero)
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        features[f'rsi_{self.period}'] = rsi
        
        # Overbought/oversold signals (vectorized boolean operations)
        features[f'rsi_{self.period}_overbought'] = (rsi > 70).astype(np.int8)
        features[f'rsi_{self.period}_oversold'] = (rsi < 30).astype(np.int8)
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('macd')
class MACDFeature(BaseFeature):
    """Moving Average Convergence Divergence indicator."""
    
    def __init__(self, name: str = 'macd', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.fast = config.get('fast', 12) if config else 12
        self.slow = config.get('slow', 26) if config else 26
        self.signal = config.get('signal', 9) if config else 9
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate MACD line
        ema_fast = data['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow, adjust=False).mean()
        features['macd'] = ema_fast - ema_slow
        
        # Calculate signal line
        features['macd_signal'] = features['macd'].ewm(span=self.signal, adjust=False).mean()
        
        # Calculate histogram
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Crossover signals
        features['macd_bullish_cross'] = ((features['macd'] > features['macd_signal']) & 
                                          (features['macd'].shift(1) <= features['macd_signal'].shift(1))).astype(int)
        features['macd_bearish_cross'] = ((features['macd'] < features['macd_signal']) & 
                                          (features['macd'].shift(1) >= features['macd_signal'].shift(1))).astype(int)
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('bollinger_bands')
class BollingerBandsFeature(BaseFeature):
    """Bollinger Bands indicator."""
    
    def __init__(self, name: str = 'bollinger_bands', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.period = config.get('period', 20) if config else 20
        self.std = config.get('std', 2) if config else 2
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute Bollinger Bands."""
        features = pd.DataFrame(index=data.index)
        
        # Middle band (SMA)
        features['bb_middle'] = data['close'].rolling(window=self.period).mean()
        
        # Standard deviation
        rolling_std = data['close'].rolling(window=self.period).std()
        
        # Upper and lower bands
        features['bb_upper'] = features['bb_middle'] + (rolling_std * self.std)
        features['bb_lower'] = features['bb_middle'] - (rolling_std * self.std)
        
        # Bandwidth
        features['bb_bandwidth'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        
        # %B indicator (position within bands)
        features['bb_pct_b'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('atr')
class ATRFeature(BaseFeature):
    """Average True Range indicator (volatility measure)."""
    
    def __init__(self, name: str = 'atr', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.period = config.get('period', 14) if config else 14
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute ATR."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        features[f'atr_{self.period}'] = true_range.rolling(window=self.period).mean()
        
        # Normalized ATR (% of price)
        features[f'atr_{self.period}_pct'] = features[f'atr_{self.period}'] / data['close']
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('volume')
class VolumeFeature(BaseFeature):
    """Volume-based features."""
    
    def __init__(self, name: str = 'volume', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.periods = config.get('periods', [5, 20]) if config else [5, 20]
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volume features."""
        features = pd.DataFrame(index=data.index)
        
        for period in self.periods:
            # Volume moving average
            features[f'volume_sma_{period}'] = data['volume'].rolling(window=period).mean()
            
            # Volume ratio
            features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_sma_{period}']
        
        # Volume-price trend
        features['vpt'] = (data['volume'] * ((data['close'] - data['close'].shift(1)) / data['close'].shift(1))).cumsum()
        
        # On-Balance Volume
        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        features['obv'] = obv
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('momentum')
class MomentumFeature(BaseFeature):
    """Momentum indicators."""
    
    def __init__(self, name: str = 'momentum', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.periods = config.get('periods', [5, 10, 20]) if config else [5, 10, 20]
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum features."""
        features = pd.DataFrame(index=data.index)
        
        for period in self.periods:
            # Price momentum
            features[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
            
            # Rate of change
            features[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / 
                                        data['close'].shift(period)) * 100
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('volatility')
class VolatilityFeature(BaseFeature):
    """Volatility measures."""
    
    def __init__(self, name: str = 'volatility', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.periods = config.get('periods', [5, 20, 60]) if config else [5, 20, 60]
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility features (vectorized)."""
        features = pd.DataFrame(index=data.index)
        
        # Calculate returns once
        returns = data['close'].pct_change()
        
        # Pre-calculate log ratios for Parkinson volatility
        hl_ratio = np.log(data['high'] / data['low'].replace(0, np.nan))
        parkinson_factor = 1.0 / (4 * np.log(2))
        sqrt_252 = np.sqrt(252)
        
        for period in self.periods:
            # Historical volatility (annualized) - vectorized
            features[f'volatility_{period}'] = returns.rolling(window=period, min_periods=period).std() * sqrt_252
            
            # Parkinson's volatility (using high-low) - vectorized
            hl_var = hl_ratio.pow(2).rolling(window=period, min_periods=period).mean()
            features[f'parkinson_vol_{period}'] = np.sqrt(hl_var * parkinson_factor) * sqrt_252
        
        self.feature_columns = list(features.columns)
        return features


@FeatureRegistry.register('price_patterns')
class PricePatternsFeature(BaseFeature):
    """Price pattern recognition features."""
    
    def __init__(self, name: str = 'price_patterns', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute price pattern features."""
        features = pd.DataFrame(index=data.index)
        
        # Candlestick body and shadow sizes
        features['body_size'] = np.abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - data[['close', 'open']].max(axis=1)
        features['lower_shadow'] = data[['close', 'open']].min(axis=1) - data['low']
        
        # Normalized by ATR
        true_range = pd.concat([
            data['high'] - data['low'],
            np.abs(data['high'] - data['close'].shift()),
            np.abs(data['low'] - data['close'].shift())
        ], axis=1).max(axis=1)
        
        atr = true_range.rolling(window=14).mean()
        features['body_size_norm'] = features['body_size'] / atr
        
        # Higher high, lower low
        features['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        features['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)
        
        # Gap detection
        features['gap_up'] = (data['low'] > data['high'].shift(1)).astype(int)
        features['gap_down'] = (data['high'] < data['low'].shift(1)).astype(int)
        
        self.feature_columns = list(features.columns)
        return features


if __name__ == "__main__":
    # Test feature registration
    from features.base_feature import FeatureRegistry
    
    print("Registered technical features:")
    for feature_name in FeatureRegistry.list_features():
        print(f"  - {feature_name}")

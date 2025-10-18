"""
Base Feature Engineering Module
Provides abstract base class for creating plug-and-play feature generators.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseFeature(ABC):
    """
    Abstract base class for feature generators.
    All custom features should inherit from this class.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature generator.
        
        Args:
            name: Feature name
            config: Configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.feature_columns = []
        
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data (date index)
            
        Returns:
            DataFrame with computed features
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_cols)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class FeatureRegistry:
    """
    Registry for managing and discovering available features.
    Enables plug-and-play architecture for adding new features.
    """
    
    _features: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a feature class.
        
        Usage:
            @FeatureRegistry.register('my_feature')
            class MyFeature(BaseFeature):
                ...
        """
        def decorator(feature_class: type):
            cls._features[name] = feature_class
            logger.info(f"Registered feature: {name}")
            return feature_class
        return decorator
    
    @classmethod
    def get_feature(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseFeature:
        """
        Get feature instance by name.
        
        Args:
            name: Feature name
            config: Feature configuration
            
        Returns:
            Feature instance
        """
        if name not in cls._features:
            raise ValueError(f"Feature '{name}' not found. Available: {list(cls._features.keys())}")
        
        feature_class = cls._features[name]
        return feature_class(name=name, config=config)
    
    @classmethod
    def list_features(cls) -> List[str]:
        """Get list of registered feature names."""
        return list(cls._features.keys())
    
    @classmethod
    def clear(cls):
        """Clear all registered features."""
        cls._features.clear()


class FeatureEngine:
    """
    Main feature engineering engine.
    Orchestrates multiple feature generators and manages feature computation.
    """
    
    def __init__(self, feature_configs: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize feature engine.
        
        Args:
            feature_configs: List of feature configurations
                Each config should have 'name' and optional parameters
        """
        self.features = []
        
        if feature_configs:
            for config in feature_configs:
                self.add_feature(config['name'], config)
    
    def add_feature(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Add a feature generator to the engine.
        
        Args:
            name: Feature name
            config: Feature configuration
        """
        try:
            feature = FeatureRegistry.get_feature(name, config)
            self.features.append(feature)
            logger.info(f"Added feature: {name}")
        except ValueError as e:
            logger.error(f"Failed to add feature '{name}': {e}")
    
    def compute_features(self, 
                        data: pd.DataFrame,
                        ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Compute all features for given OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            ticker: Optional ticker symbol for logging
            
        Returns:
            DataFrame with all computed features
        """
        if data.empty:
            logger.warning(f"Empty data for {ticker}")
            return pd.DataFrame()
        
        # Start with original data
        result = data.copy()
        
        # Compute each feature
        for feature in self.features:
            try:
                logger.debug(f"Computing {feature.name} for {ticker}...")
                feature_data = feature.compute(result)
                
                # Merge features
                if not feature_data.empty:
                    result = result.join(feature_data, how='left')
                    
            except Exception as e:
                logger.error(f"Error computing {feature.name} for {ticker}: {e}")
        
        return result
    
    def compute_features_batch(self, 
                              data_dict: Dict[str, pd.DataFrame],
                              save_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Compute features for multiple tickers.
        
        Args:
            data_dict: Dictionary mapping ticker to OHLCV DataFrame
            save_dir: Optional directory to save feature CSVs
            
        Returns:
            Dictionary mapping ticker to feature DataFrame
        """
        from pathlib import Path
        
        results = {}
        
        for ticker, data in data_dict.items():
            logger.info(f"Computing features for {ticker}...")
            features = self.compute_features(data, ticker)
            results[ticker] = features
            
            # Save to CSV if directory provided
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                filepath = save_path / f"{ticker}_features.csv"
                features.reset_index().to_csv(filepath, index=False)
                logger.info(f"Saved features to {filepath}")
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature column names."""
        all_names = []
        for feature in self.features:
            all_names.extend(feature.get_feature_names())
        return all_names
    
    def __repr__(self) -> str:
        feature_names = [f.name for f in self.features]
        return f"FeatureEngine(features={feature_names})"


if __name__ == "__main__":
    # Example usage
    print(f"Available features: {FeatureRegistry.list_features()}")
    
    # Create feature engine
    engine = FeatureEngine()
    print(f"Feature engine: {engine}")

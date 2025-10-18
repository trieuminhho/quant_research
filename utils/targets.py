"""
Target Variable Creation
Generate forward-looking target variables for ML model training.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def create_forward_returns(data: pd.DataFrame,
                          horizons: list = [1, 5, 21],
                          return_type: str = 'log') -> pd.DataFrame:
    """
    Create forward-looking return targets.
    
    Args:
        data: DataFrame with 'close' prices (date-indexed)
        horizons: List of forward horizons in days
        return_type: 'log' or 'simple'
        
    Returns:
        DataFrame with forward return columns
    """
    if 'close' not in data.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    targets = pd.DataFrame(index=data.index)
    close_prices = data['close']
    
    if return_type == 'log':
        # Pre-compute log for efficiency
        log_close = np.log(close_prices)
        for horizon in horizons:
            targets[f'forward_return_{horizon}d'] = log_close.shift(-horizon) - log_close
    else:  # simple
        for horizon in horizons:
            targets[f'forward_return_{horizon}d'] = close_prices.pct_change(-horizon)
    
    return targets


def create_target_labels(data: pd.DataFrame,
                        horizon: int = 21,
                        method: str = 'direction',
                        quantiles: Optional[list] = None) -> pd.Series:
    """
    Create categorical target labels for classification.
    
    Args:
        data: DataFrame with price data
        horizon: Forward horizon in days
        method: 'direction' (up/down) or 'quantile' (multi-class)
        quantiles: Quantile thresholds if using quantile method
        
    Returns:
        Series with target labels
    """
    # Calculate forward returns
    forward_return = data['close'].shift(-horizon) / data['close'] - 1
    
    if method == 'direction':
        # Binary: 1 = up, 0 = down
        labels = (forward_return > 0).astype(int)
        
    elif method == 'quantile':
        # Multi-class based on return quantiles
        if quantiles is None:
            quantiles = [0.2, 0.4, 0.6, 0.8]
        
        labels = pd.cut(
            forward_return,
            bins=[-np.inf] + list(forward_return.quantile(quantiles)) + [np.inf],
            labels=range(len(quantiles) + 1)
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels


def add_targets_to_features(features_path: str,
                           ohlcv_data: pd.DataFrame,
                           horizons: list = [21],
                           return_type: str = 'log') -> pd.DataFrame:
    """
    Add target variables to existing features CSV.
    
    Args:
        features_path: Path to features CSV
        ohlcv_data: OHLCV data for creating targets (date-indexed)
        horizons: Forward return horizons
        return_type: Type of returns ('log' or 'simple')
        
    Returns:
        Combined DataFrame with features and targets
    """
    # Load features with efficient parsing
    features = pd.read_csv(features_path, parse_dates=['date'], index_col='date')
    
    # Create targets
    targets = create_forward_returns(ohlcv_data, horizons, return_type)
    
    # Merge efficiently using join
    combined = features.join(targets, how='left')
    
    return combined


if __name__ == "__main__":
    # Example
    print("Target creation utilities loaded")

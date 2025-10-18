"""
Configuration Management System
Centralized configuration loading and management for the QuantSearch platform.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


class Config:
    """
    Centralized configuration manager for the QuantSearch platform.
    Loads configuration from YAML file and provides easy access to settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file. If None, searches in project root.
        """
        if config_path is None:
            # Search for config.yaml in project root
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config.yaml"
        
        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent
        self._config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate configuration and create necessary directories."""
        # Create directories if they don't exist
        paths_to_create = [
            self.get_path('raw_data'),
            self.get_path('processed_features'),
            self.get_path('processed_predictions'),
            self.get_path('strategy_templates'),
            self.get_path('saved_models'),
            self.get_path('backtest_results'),
            self.project_root / 'logs'
        ]
        
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, key: str) -> Path:
        """
        Get absolute path from configuration.
        
        Args:
            key: Path key from config (e.g., 'raw_data', 'saved_models')
            
        Returns:
            Absolute Path object
        """
        relative_path = self._config['paths'][key]
        return self.project_root / relative_path
    
    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            *keys: Nested keys to traverse (e.g., 'data', 'start_date')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, *keys, value):
        """
        Set configuration value using dot notation.
        
        Args:
            *keys: Nested keys to traverse
            value: Value to set
        """
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    @property
    def data_start_date(self) -> str:
        """Get data start date."""
        return self.get('data', 'start_date')
    
    @property
    def data_end_date(self) -> Optional[str]:
        """Get data end date (None = today)."""
        end_date = self.get('data', 'end_date')
        if end_date is None:
            return datetime.now().strftime('%Y-%m-%d')
        return end_date
    
    @property
    def model_type(self) -> str:
        """Get default model type."""
        return self.get('models', 'default_model', default='lightgbm')
    
    @property
    def model_params(self) -> Dict[str, Any]:
        """Get model parameters for the default model."""
        return self.get('models', self.model_type, default={})
    
    @property
    def max_positions(self) -> int:
        """Get maximum number of portfolio positions."""
        return self.get('strategy', 'portfolio', 'max_positions', default=20)
    
    @property
    def rebalance_frequency(self) -> str:
        """Get portfolio rebalance frequency."""
        return self.get('strategy', 'rebalance_frequency', default='monthly')
    
    @property
    def initial_capital(self) -> float:
        """Get initial backtest capital."""
        return self.get('backtest', 'initial_capital', default=1000000)
    
    @property
    def commission_pct(self) -> float:
        """Get commission percentage."""
        return self.get('backtest', 'costs', 'commission_pct', default=0.001)
    
    @property
    def slippage_bps(self) -> float:
        """Get slippage in basis points."""
        return self.get('backtest', 'costs', 'slippage_bps', default=5)
    
    def save(self, path: Optional[str] = None):
        """
        Save current configuration to YAML file.
        
        Args:
            path: Optional path to save to. If None, overwrites original file.
        """
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"Config(config_path={self.config_path})"


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def reload_config(config_path: Optional[str] = None):
    """
    Reload configuration from file.
    
    Args:
        config_path: Optional path to config file
    """
    global _global_config
    _global_config = Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Configuration loaded from: {config.config_path}")
    print(f"Project root: {config.project_root}")
    print(f"Data start date: {config.data_start_date}")
    print(f"Model type: {config.model_type}")
    print(f"Max positions: {config.max_positions}")

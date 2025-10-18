"""
Model Training Module
Implements ML model training with walk-forward validation for time series data.
Supports LightGBM, XGBoost, and other ML models with CSV output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import joblib

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains ML models using walk-forward expanding window validation.
    Designed for time series prediction with no lookahead bias.
    """
    
    def __init__(self,
                 model_type: str = 'lightgbm',
                 model_params: Optional[Dict[str, Any]] = None,
                 output_dir: str = "models/saved"):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost', 'catboost')
            model_params: Model hyperparameters
            output_dir: Directory to save trained models
        """
        self.model_type = model_type
        self.model_params = model_params or self._get_default_params()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_importance = None
        self.training_history = []
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the model type."""
        if self.model_type == 'lightgbm':
            return {
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        elif self.model_type == 'xgboost':
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            return {}
    
    def _create_model(self):
        """Create a new model instance."""
        if self.model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMRegressor(**self.model_params)
        
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBRegressor(**self.model_params)
        
        elif self.model_type == 'catboost':
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**self.model_params, verbose=False)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_features_target(self,
                               data: pd.DataFrame,
                               target_col: str = 'target',
                               feature_cols: Optional[List[str]] = None,
                               drop_na: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from DataFrame.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature columns (None = all except target)
            drop_na: Whether to drop rows with missing values
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Select feature columns
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
        else:
            # Validate feature columns exist
            missing_cols = set(feature_cols) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
        
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Drop missing values efficiently
        if drop_na:
            # Use loc for efficiency - find valid rows once
            valid_mask = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]
        
        return X, y
    
    def walk_forward_split(self,
                          data: pd.DataFrame,
                          train_period_months: int = 36,
                          test_period_months: int = 1,
                          expanding_window: bool = True,
                          min_train_samples: int = 1000) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward train/test splits.
        
        Args:
            data: DataFrame with date index
            train_period_months: Training period in months
            test_period_months: Test period in months
            expanding_window: If True, use expanding window; if False, use rolling window
            min_train_samples: Minimum training samples required
            
        Returns:
            List of (train_data, test_data) tuples
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        splits = []
        
        # Get date range
        start_date = data.index.min()
        end_date = data.index.max()
        
        # Initial training end date
        train_end = start_date + pd.DateOffset(months=train_period_months)
        
        while train_end < end_date:
            # Test period
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_period_months)
            
            if test_end > end_date:
                test_end = end_date
            
            # Training period
            if expanding_window:
                train_start = start_date
            else:
                train_start = train_end - pd.DateOffset(months=train_period_months)
            
            # Extract data
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            # Check minimum samples
            if len(train_data) >= min_train_samples and len(test_data) > 0:
                splits.append((train_data, test_data))
                logger.info(f"Split: Train {train_start.date()} to {train_end.date()} ({len(train_data)} samples), "
                          f"Test {test_start.date()} to {test_end.date()} ({len(test_data)} samples)")
            
            # Move forward by test period
            train_end = test_end
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def train(self,
             X_train: pd.DataFrame,
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None) -> Any:
        """
        Train model on given data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Trained model
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        
        # Create model
        self.model = self._create_model()
        
        # Train with validation set if provided
        if X_val is not None and y_val is not None:
            if self.model_type in ['lightgbm', 'xgboost']:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info("Model training complete")
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def evaluate(self,
                y_true: pd.Series,
                y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        from scipy.stats import spearmanr
        
        # Convert to numpy arrays for efficiency
        y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
        
        # Remove NaN values efficiently
        mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred))
        y_true_clean = y_true_arr[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {}
        
        # Calculate metrics (vectorized where possible)
        residuals = y_true_clean - y_pred_clean
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Spearman correlation (rank correlation)
        spearman_corr, _ = spearmanr(y_true_clean, y_pred_clean)
        
        # Information Coefficient (Spearman IC)
        ic = spearman_corr
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'spearman': spearman_corr,
            'ic': ic,
            'n_samples': len(y_true_clean)
        }
        
        return metrics
    
    def walk_forward_train(self,
                          data: pd.DataFrame,
                          target_col: str = 'target',
                          feature_cols: Optional[List[str]] = None,
                          train_period_months: int = 36,
                          test_period_months: int = 1,
                          expanding_window: bool = True,
                          save_predictions: bool = True,
                          predictions_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Perform walk-forward training and generate predictions.
        
        Args:
            data: DataFrame with features and target
            target_col: Target column name
            feature_cols: List of feature columns
            train_period_months: Training window in months
            test_period_months: Test window in months
            expanding_window: Use expanding window
            save_predictions: Save predictions to CSV
            predictions_dir: Directory to save predictions
            
        Returns:
            DataFrame with predictions and evaluation metrics
        """
        # Prepare data
        X, y = self.prepare_features_target(data, target_col, feature_cols)
        
        # Combine back for splitting
        data_prepared = X.copy()
        data_prepared[target_col] = y
        
        # Generate splits
        splits = self.walk_forward_split(
            data_prepared,
            train_period_months=train_period_months,
            test_period_months=test_period_months,
            expanding_window=expanding_window
        )
        
        # Store all predictions
        all_predictions = []
        all_metrics = []
        
        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}...")
            
            # Prepare split data
            X_train, y_train = self.prepare_features_target(train_data, target_col, feature_cols)
            X_test, y_test = self.prepare_features_target(test_data, target_col, feature_cols)
            
            # Train model
            self.train(X_train, y_train)
            
            # Predict
            y_pred = self.predict(X_test)
            
            # Evaluate
            metrics = self.evaluate(y_test, y_pred)
            metrics['split'] = i
            metrics['train_end'] = train_data.index.max()
            metrics['test_start'] = test_data.index.min()
            metrics['test_end'] = test_data.index.max()
            all_metrics.append(metrics)
            
            logger.info(f"  Metrics: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}, IC={metrics['ic']:.4f}")
            
            # Store predictions
            pred_df = pd.DataFrame({
                'date': X_test.index,
                'actual': y_test.values,
                'predicted': y_pred,
                'split': i
            })
            all_predictions.append(pred_df)
        
        # Combine predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        # Save predictions
        if save_predictions and predictions_dir:
            pred_path = Path(predictions_dir)
            pred_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pred_file = pred_path / f"predictions_{self.model_type}_{timestamp}.csv"
            predictions_df.to_csv(pred_file, index=False)
            logger.info(f"Saved predictions to {pred_file}")
        
        # Save metrics summary
        metrics_df = pd.DataFrame(all_metrics)
        self.training_history = metrics_df
        
        return predictions_df
    
    def save_model(self, filepath: Optional[str] = None, ticker: Optional[str] = None):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model. If None, auto-generates name.
            ticker: Optional ticker symbol for filename
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ticker_str = f"{ticker}_" if ticker else ""
            filename = f"{ticker_str}{self.model_type}_model_{timestamp}.pkl"
            filepath = self.output_dir / filename
        
        joblib.dump(self.model, filepath)
        logger.info(f"Saved model to {filepath}")
        
        # Save feature importance if available
        if self.feature_importance is not None:
            importance_file = str(filepath).replace('.pkl', '_importance.csv')
            self.feature_importance.to_csv(importance_file, index=False)
            logger.info(f"Saved feature importance to {importance_file}")
    
    def load_model(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        self.model = joblib.load(filepath)
        logger.info(f"Loaded model from {filepath}")
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of training history."""
        if not self.training_history:
            return pd.DataFrame()
        
        return self.training_history


if __name__ == "__main__":
    # Example usage
    print("ModelTrainer initialized")
    
    # Create sample data
    dates = pd.date_range('2015-01-01', '2023-12-31', freq='D')
    n = len(dates)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'target': np.random.randn(n)
    })
    sample_data.set_index('date', inplace=True)
    
    # Train model
    trainer = ModelTrainer(model_type='lightgbm')
    predictions = trainer.walk_forward_train(
        sample_data,
        target_col='target',
        train_period_months=36,
        test_period_months=1
    )
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Training summary:\n{trainer.get_training_summary()}")

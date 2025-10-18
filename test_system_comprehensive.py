"""
Comprehensive System QA Test
Tests entire QuantSearch pipeline from data ingestion to backtesting.
Validates all DataFrames, variables, and outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from data.ingestion import DataIngestion
from data.validation import DataValidator
from utils.data_loader import DataLoader, UniverseFilter
from utils.config import get_config
from features.base_feature import FeatureEngine, FeatureRegistry
from features.technical_indicators import *
from models.trainer import ModelTrainer
from strategies.base_strategy import MLStrategy
from backtest.engine import BacktestEngine
from backtest.portfolio import PortfolioConstructor


class SystemQATester:
    """Comprehensive system tester."""
    
    def __init__(self):
        """Initialize tester."""
        self.test_results = []
        self.test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.passed = 0
        self.failed = 0
        
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def validate_dataframe(self, df: pd.DataFrame, name: str, 
                          required_cols: List[str] = None,
                          required_index_type: type = None) -> bool:
        """
        Validate DataFrame structure.
        
        Args:
            df: DataFrame to validate
            name: Name for logging
            required_cols: Required column names
            required_index_type: Required index type
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check not empty
            if df is None or df.empty:
                self.log_test(f"DataFrame {name}", False, "DataFrame is empty")
                return False
            
            # Check required columns
            if required_cols:
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    self.log_test(f"DataFrame {name} columns", False, 
                                f"Missing columns: {missing_cols}")
                    return False
            
            # Check index type
            if required_index_type and not isinstance(df.index, required_index_type):
                self.log_test(f"DataFrame {name} index", False,
                            f"Expected {required_index_type}, got {type(df.index)}")
                return False
            
            # Check for NaN in index
            if df.index.hasnans:
                self.log_test(f"DataFrame {name} index", False, "Index contains NaN values")
                return False
            
            # Check for duplicate index
            if df.index.duplicated().any():
                self.log_test(f"DataFrame {name} index", False, "Index contains duplicates")
                return False
            
            self.log_test(f"DataFrame {name}", True, 
                        f"Shape: {df.shape}, Columns: {len(df.columns)}")
            return True
            
        except Exception as e:
            self.log_test(f"DataFrame {name}", False, f"Error: {str(e)}")
            return False
    
    def test_1_data_ingestion(self) -> bool:
        """Test data ingestion module."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Data Ingestion")
        logger.info("="*80)
        
        try:
            ingestion = DataIngestion(output_dir="data/raw/ohlcv")
            
            # Test ticker fetching
            tickers = ingestion.get_sp500_tickers()
            self.log_test("Fetch S&P 500 tickers", len(tickers) > 0,
                         f"Got {len(tickers)} tickers")
            
            # Test downloading single ticker (use existing data if available)
            loader = DataLoader()
            existing_tickers = loader.get_available_tickers()
            
            if len(existing_tickers) >= 3:
                self.log_test("Data already available", True,
                            f"Using {len(existing_tickers)} existing tickers")
                return True
            else:
                # Download test tickers
                logger.info(f"Downloading test tickers: {self.test_tickers}")
                results = ingestion.download_all_tickers(
                    tickers=self.test_tickers,
                    max_workers=3
                )
                
                success_count = sum(1 for v in results.values() if v)
                self.log_test("Download tickers", success_count > 0,
                            f"{success_count}/{len(self.test_tickers)} successful")
                
                return success_count > 0
                
        except Exception as e:
            self.log_test("Data ingestion", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def test_2_data_loading(self) -> Tuple[bool, Dict[str, pd.DataFrame]]:
        """Test data loading module."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Data Loading")
        logger.info("="*80)
        
        try:
            loader = DataLoader()
            
            # Test get available tickers
            tickers = loader.get_available_tickers()
            self.log_test("Get available tickers", len(tickers) > 0,
                         f"Found {len(tickers)} tickers")
            
            if len(tickers) == 0:
                return False, {}
            
            # Use first 3 tickers for testing
            test_tickers = tickers[:3]
            data_dict = {}
            
            for ticker in test_tickers:
                data = loader.load_ticker_ohlcv(ticker)
                
                if data is not None:
                    # Validate DataFrame structure
                    is_valid = self.validate_dataframe(
                        data,
                        f"{ticker} OHLCV",
                        required_cols=['open', 'high', 'low', 'close', 'volume'],
                        required_index_type=pd.DatetimeIndex
                    )
                    
                    if is_valid:
                        # Check data quality
                        has_negatives = (data[['open', 'high', 'low', 'close']] < 0).any().any()
                        has_zeros = (data[['open', 'high', 'low', 'close']] == 0).any().any()
                        
                        if has_negatives or has_zeros:
                            self.log_test(f"{ticker} data quality", False,
                                        "Contains negative or zero prices")
                        else:
                            self.log_test(f"{ticker} data quality", True,
                                        f"No anomalies, {len(data)} records")
                            data_dict[ticker] = data
            
            return len(data_dict) > 0, data_dict
            
        except Exception as e:
            self.log_test("Data loading", False, f"Error: {str(e)}")
            logger.exception(e)
            return False, {}
    
    def test_3_data_validation(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """Test data validation module."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Data Validation")
        logger.info("="*80)
        
        try:
            validator = DataValidator()
            
            # Test validation for one ticker
            if not data_dict:
                self.log_test("Data validation", False, "No data to validate")
                return False
            
            ticker = list(data_dict.keys())[0]
            data = data_dict[ticker]
            
            # Validate price data
            results = validator.validate_price_data(ticker, data)
            
            # Check validation results structure
            required_keys = ['ticker', 'is_valid', 'confidence_score', 'data_quality']
            has_all_keys = all(key in results for key in required_keys)
            
            self.log_test("Validation results structure", has_all_keys,
                         f"Has all required keys: {required_keys}")
            
            self.log_test(f"{ticker} validation", results['is_valid'],
                         f"Quality: {results['data_quality']}, "
                         f"Confidence: {results['confidence_score']:.1f}%")
            
            return True
            
        except Exception as e:
            self.log_test("Data validation", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def test_4_feature_engineering(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict[str, pd.DataFrame]]:
        """Test feature engineering module."""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Feature Engineering")
        logger.info("="*80)
        
        try:
            # Check registered features
            features_list = FeatureRegistry.list_features()
            self.log_test("Feature registry", len(features_list) > 0,
                         f"Registered features: {', '.join(features_list)}")
            
            # Create feature engine
            engine = FeatureEngine()
            engine.add_feature('returns')
            engine.add_feature('sma')
            engine.add_feature('rsi')
            engine.add_feature('volatility')
            
            self.log_test("Feature engine creation", True,
                         f"Added {len(engine.features)} features")
            
            # Compute features for test tickers
            feature_dict = {}
            
            for ticker, data in data_dict.items():
                features = engine.compute_features(data, ticker)
                
                # Validate features DataFrame
                is_valid = self.validate_dataframe(
                    features,
                    f"{ticker} features",
                    required_index_type=pd.DatetimeIndex
                )
                
                if is_valid:
                    # Check that features were actually computed
                    feature_cols = [col for col in features.columns 
                                  if col not in ['open', 'high', 'low', 'close', 'volume', 'ticker']]
                    
                    self.log_test(f"{ticker} feature computation", len(feature_cols) > 0,
                                f"Computed {len(feature_cols)} features")
                    
                    # Check for excessive NaN values
                    nan_pct = features[feature_cols].isnull().sum().sum() / (len(features) * len(feature_cols))
                    self.log_test(f"{ticker} feature quality", nan_pct < 0.3,
                                f"NaN percentage: {nan_pct*100:.1f}%")
                    
                    feature_dict[ticker] = features
            
            return len(feature_dict) > 0, feature_dict
            
        except Exception as e:
            self.log_test("Feature engineering", False, f"Error: {str(e)}")
            logger.exception(e)
            return False, {}
    
    def test_5_target_creation(self, feature_dict: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict[str, pd.DataFrame]]:
        """Test target variable creation."""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Target Creation")
        logger.info("="*80)
        
        try:
            from utils.targets import create_forward_return_target
            
            target_dict = {}
            
            for ticker, features in feature_dict.items():
                # Create target (forward return)
                target = features['close'].pct_change(5).shift(-5)  # 5-day forward return
                
                # Add target to features
                features_with_target = features.copy()
                features_with_target['target'] = target
                
                # Validate
                has_target = 'target' in features_with_target.columns
                self.log_test(f"{ticker} target creation", has_target,
                            f"Created forward return target")
                
                if has_target:
                    # Check target distribution
                    target_clean = target.dropna()
                    if len(target_clean) > 0:
                        mean_return = target_clean.mean()
                        std_return = target_clean.std()
                        
                        self.log_test(f"{ticker} target distribution", True,
                                    f"Mean: {mean_return*100:.2f}%, Std: {std_return*100:.2f}%")
                        
                        target_dict[ticker] = features_with_target
            
            return len(target_dict) > 0, target_dict
            
        except Exception as e:
            self.log_test("Target creation", False, f"Error: {str(e)}")
            logger.exception(e)
            return False, {}
    
    def test_6_model_training(self, data_with_targets: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict[str, pd.DataFrame]]:
        """Test ML model training module."""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Model Training")
        logger.info("="*80)
        
        try:
            # Test with one ticker
            ticker = list(data_with_targets.keys())[0]
            data = data_with_targets[ticker]
            
            # Prepare features (select only technical features, not OHLCV)
            feature_cols = [col for col in data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'ticker', 'target']]
            
            self.log_test("Feature selection", len(feature_cols) > 0,
                         f"Selected {len(feature_cols)} features for training")
            
            # Create trainer
            trainer = ModelTrainer(
                model_type='lightgbm',
                output_dir='models/saved'
            )
            
            self.log_test("Model trainer initialization", True,
                         f"Model type: {trainer.model_type}")
            
            # Prepare data for training
            X, y = trainer.prepare_features_target(
                data,
                target_col='target',
                feature_cols=feature_cols
            )
            
            self.log_test("Data preparation", len(X) > 0 and len(y) > 0,
                         f"X shape: {X.shape}, y shape: {y.shape}")
            
            # Generate walk-forward splits (use shorter periods for testing)
            splits = trainer.walk_forward_split(
                pd.concat([X, y], axis=1),
                train_period_months=12,
                test_period_months=1,
                min_train_samples=100
            )
            
            self.log_test("Walk-forward splits", len(splits) > 0,
                         f"Generated {len(splits)} splits")
            
            if len(splits) == 0:
                return False, {}
            
            # Train on first split only (for speed)
            train_data, test_data = splits[0]
            
            X_train, y_train = trainer.prepare_features_target(
                train_data,
                target_col='target',
                feature_cols=feature_cols
            )
            
            X_test, y_test = trainer.prepare_features_target(
                test_data,
                target_col='target',
                feature_cols=feature_cols
            )
            
            # Train model
            trainer.train(X_train, y_train)
            
            self.log_test("Model training", trainer.model is not None,
                         f"Trained on {len(X_train)} samples")
            
            # Make predictions
            y_pred = trainer.predict(X_test)
            
            self.log_test("Prediction generation", len(y_pred) == len(y_test),
                         f"Generated {len(y_pred)} predictions")
            
            # Evaluate
            metrics = trainer.evaluate(y_test, y_pred)
            
            self.log_test("Model evaluation", 'mae' in metrics and 'ic' in metrics,
                         f"MAE: {metrics.get('mae', 0):.4f}, IC: {metrics.get('ic', 0):.4f}")
            
            # Create predictions DataFrame for strategy
            predictions_df = pd.DataFrame({
                'predicted_return': y_pred,
                'confidence': 0.5  # Placeholder
            }, index=X_test.index)
            
            # Add volatility (reuse from features)
            if 'volatility_20' in data.columns:
                predictions_df['volatility'] = data.loc[X_test.index, 'volatility_20']
            
            prediction_dict = {ticker: predictions_df}
            
            return True, prediction_dict
            
        except Exception as e:
            self.log_test("Model training", False, f"Error: {str(e)}")
            logger.exception(e)
            return False, {}
    
    def test_7_strategy_signals(self, predictions: Dict[str, pd.DataFrame],
                               data_dict: Dict[str, pd.DataFrame]) -> bool:
        """Test strategy signal generation."""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: Strategy Signal Generation")
        logger.info("="*80)
        
        try:
            ticker = list(predictions.keys())[0]
            pred_df = predictions[ticker]
            
            # Create strategy
            strategy = MLStrategy(
                name='Test ML Strategy',
                universe=[ticker],
                model_predictions=predictions,
                config={
                    'max_positions': 10,
                    'rebalance_frequency': 'monthly'
                }
            )
            
            self.log_test("Strategy initialization", True,
                         f"Created {strategy.name}")
            
            # Generate signals for a test date
            test_date = pred_df.index[len(pred_df)//2]  # Use middle date
            
            signals = strategy.generate_signals(data_dict, test_date)
            
            self.log_test("Signal generation", not signals.empty,
                         f"Generated {len(signals)} signals")
            
            if not signals.empty:
                # Validate signals structure
                required_cols = ['ticker', 'signal', 'predicted_return']
                has_required = all(col in signals.columns for col in required_cols)
                
                self.log_test("Signal structure", has_required,
                             f"Columns: {list(signals.columns)}")
                
                # Check signal values
                valid_signals = signals['signal'].isin([-1, 0, 1]).all()
                self.log_test("Signal values", valid_signals,
                             f"All signals in [-1, 0, 1]")
            
            return True
            
        except Exception as e:
            self.log_test("Strategy signals", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def test_8_portfolio_construction(self, predictions: Dict[str, pd.DataFrame]) -> bool:
        """Test portfolio construction."""
        logger.info("\n" + "="*80)
        logger.info("TEST 8: Portfolio Construction")
        logger.info("="*80)
        
        try:
            ticker = list(predictions.keys())[0]
            pred_df = predictions[ticker]
            
            # Create strategy
            strategy = MLStrategy(
                name='Test ML Strategy',
                universe=[ticker],
                model_predictions=predictions,
                config={
                    'max_positions': 10,
                    'use_risk_weighting': True
                }
            )
            
            # Generate signals
            test_date = pred_df.index[len(pred_df)//2]
            signals = strategy.generate_signals({}, test_date)
            
            if signals.empty:
                self.log_test("Portfolio construction", False, "No signals generated")
                return False
            
            # Select portfolio
            portfolio = strategy.select_portfolio(signals, {}, test_date)
            
            self.log_test("Portfolio selection", len(portfolio) > 0,
                         f"Selected {len(portfolio)} positions")
            
            # Check weights sum to 1
            if portfolio:
                total_weight = sum(portfolio.values())
                weights_valid = abs(total_weight - 1.0) < 0.01
                
                self.log_test("Portfolio weights", weights_valid,
                             f"Total weight: {total_weight:.4f}")
            
            return True
            
        except Exception as e:
            self.log_test("Portfolio construction", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def test_9_config_system(self) -> bool:
        """Test configuration system."""
        logger.info("\n" + "="*80)
        logger.info("TEST 9: Configuration System")
        logger.info("="*80)
        
        try:
            config = get_config()
            
            self.log_test("Config loading", config is not None,
                         "Configuration loaded successfully")
            
            # Test path resolution
            data_path = config.get_path('data_root')
            self.log_test("Path resolution", Path(data_path).exists(),
                         f"Data path: {data_path}")
            
            return True
            
        except Exception as e:
            self.log_test("Config system", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPREHENSIVE SYSTEM QA TEST")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # Test 1: Data Ingestion
        test_1_pass = self.test_1_data_ingestion()
        
        # Test 2: Data Loading
        test_2_pass, data_dict = self.test_2_data_loading()
        
        if not test_2_pass or not data_dict:
            logger.error("Cannot continue without data. Please ensure data is downloaded.")
            self.print_summary()
            return
        
        # Test 3: Data Validation
        self.test_3_data_validation(data_dict)
        
        # Test 4: Feature Engineering
        test_4_pass, feature_dict = self.test_4_feature_engineering(data_dict)
        
        if test_4_pass and feature_dict:
            # Test 5: Target Creation
            test_5_pass, target_dict = self.test_5_target_creation(feature_dict)
            
            if test_5_pass and target_dict:
                # Test 6: Model Training
                test_6_pass, prediction_dict = self.test_6_model_training(target_dict)
                
                if test_6_pass and prediction_dict:
                    # Test 7: Strategy Signals
                    self.test_7_strategy_signals(prediction_dict, data_dict)
                    
                    # Test 8: Portfolio Construction
                    self.test_8_portfolio_construction(prediction_dict)
        
        # Test 9: Config System
        self.test_9_config_system()
        
        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total tests: {self.passed + self.failed}")
        logger.info(f"Passed: {self.passed} ✅")
        logger.info(f"Failed: {self.failed} ❌")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info("="*80)
        
        # Save detailed results
        self.save_results()
    
    def save_results(self):
        """Save test results to CSV."""
        results_df = pd.DataFrame(self.test_results)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"test_results_{timestamp}.csv"
        
        results_df.to_csv(filepath, index=False)
        logger.info(f"\n📊 Detailed results saved to: {filepath}")
    
    def print_summary(self):
        """Print summary without running more tests."""
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total tests: {self.passed + self.failed}")
        logger.info(f"Passed: {self.passed} ✅")
        logger.info(f"Failed: {self.failed} ❌")
        logger.info("="*80)


if __name__ == "__main__":
    tester = SystemQATester()
    tester.run_all_tests()

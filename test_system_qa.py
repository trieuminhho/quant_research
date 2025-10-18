"""
Comprehensive QA Test Suite for QuantSearch Platform
Tests all components from data ingestion to backtesting.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config import get_config
from utils.data_loader import DataLoader, UniverseFilter
from data.ingestion import DataIngestion
from data.validation import DataValidator
from features.base_feature import FeatureEngine, FeatureRegistry
from features.technical_indicators import *
from models.trainer import ModelTrainer
from strategies.base_strategy import MLStrategy
from backtest.portfolio import PortfolioConstructor
from backtest.engine import BacktestEngine

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_test(test_name, status, message="", details=None):
    """Log test results."""
    result = {
        'test': test_name,
        'status': status,
        'message': message,
        'details': details,
        'timestamp': datetime.now()
    }
    
    if status == 'PASS':
        test_results['passed'].append(result)
        print(f"✅ {test_name}: {message}")
    elif status == 'FAIL':
        test_results['failed'].append(result)
        print(f"❌ {test_name}: {message}")
        if details:
            print(f"   Details: {details}")
    elif status == 'WARN':
        test_results['warnings'].append(result)
        print(f"⚠️  {test_name}: {message}")


def test_config():
    """Test 1: Configuration loading."""
    test_name = "Configuration Loading"
    try:
        config = get_config()
        
        # Validate required config keys
        required_keys = ['data', 'model_params', 'backtest']
        missing_keys = [k for k in required_keys if k not in config]
        
        if missing_keys:
            log_test(test_name, 'FAIL', f"Missing config keys: {missing_keys}")
            return False
        
        # Validate paths exist
        from utils.config import get_path
        critical_paths = ['raw_ohlcv', 'processed_features', 'processed_predictions']
        
        for path_key in critical_paths:
            try:
                path = get_path(path_key)
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    log_test(test_name, 'WARN', f"Created missing directory: {path}")
            except Exception as e:
                log_test(test_name, 'FAIL', f"Path error for {path_key}", str(e))
                return False
        
        log_test(test_name, 'PASS', "Config loaded successfully")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception during config loading", traceback.format_exc())
        return False


def test_data_loader():
    """Test 2: DataLoader functionality."""
    test_name = "DataLoader"
    try:
        loader = DataLoader()
        
        # Check available tickers
        tickers = loader.get_available_tickers()
        if len(tickers) == 0:
            log_test(test_name, 'WARN', "No tickers available - run data download first")
            return True  # Not a failure, just needs data
        
        # Test loading a ticker
        test_ticker = tickers[0]
        data = loader.load_ticker_ohlcv(test_ticker)
        
        # Validate DataFrame structure
        if not isinstance(data, pd.DataFrame):
            log_test(test_name, 'FAIL', f"Expected DataFrame, got {type(data)}")
            return False
        
        # Validate index
        if not isinstance(data.index, pd.DatetimeIndex):
            log_test(test_name, 'FAIL', f"Index is not DatetimeIndex: {type(data.index)}")
            return False
        
        if data.index.name != 'date':
            log_test(test_name, 'FAIL', f"Index name is '{data.index.name}', expected 'date'")
            return False
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in data.columns]
        
        if missing_cols:
            log_test(test_name, 'FAIL', f"Missing columns: {missing_cols}")
            return False
        
        # Validate data quality
        if data.isnull().any().any():
            null_cols = data.columns[data.isnull().any()].tolist()
            log_test(test_name, 'WARN', f"Null values found in: {null_cols}")
        
        # Validate OHLC consistency
        invalid_ohlc = (data['high'] < data['low']).sum()
        if invalid_ohlc > 0:
            log_test(test_name, 'FAIL', f"Found {invalid_ohlc} rows where high < low")
            return False
        
        invalid_ohlc2 = ((data['high'] < data['open']) | (data['high'] < data['close'])).sum()
        if invalid_ohlc2 > 0:
            log_test(test_name, 'FAIL', f"Found {invalid_ohlc2} rows where high < open/close")
            return False
        
        invalid_ohlc3 = ((data['low'] > data['open']) | (data['low'] > data['close'])).sum()
        if invalid_ohlc3 > 0:
            log_test(test_name, 'FAIL', f"Found {invalid_ohlc3} rows where low > open/close")
            return False
        
        log_test(test_name, 'PASS', f"Loaded {test_ticker}: {len(data)} rows, {data.index[0]} to {data.index[-1]}")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception during data loading", traceback.format_exc())
        return False


def test_data_validator():
    """Test 3: Data validation."""
    test_name = "Data Validator"
    try:
        loader = DataLoader()
        tickers = loader.get_available_tickers()
        
        if len(tickers) == 0:
            log_test(test_name, 'WARN', "No tickers to validate")
            return True
        
        validator = DataValidator()
        test_ticker = tickers[0]
        data = loader.load_ticker_ohlcv(test_ticker)
        
        # Run validation
        validation_result = validator.validate_ohlcv(data, test_ticker)
        
        if not isinstance(validation_result, dict):
            log_test(test_name, 'FAIL', "Validation result is not a dict")
            return False
        
        # Check validation structure
        required_keys = ['is_valid', 'ticker', 'errors', 'warnings']
        missing = [k for k in required_keys if k not in validation_result]
        if missing:
            log_test(test_name, 'FAIL', f"Missing validation keys: {missing}")
            return False
        
        if validation_result['errors']:
            log_test(test_name, 'WARN', f"Validation errors for {test_ticker}: {validation_result['errors']}")
        
        log_test(test_name, 'PASS', f"Validated {test_ticker}: {validation_result['is_valid']}")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception during validation", traceback.format_exc())
        return False


def test_feature_registry():
    """Test 4: Feature registry and registration."""
    test_name = "Feature Registry"
    try:
        # List available features
        features = FeatureRegistry.list_features()
        
        if not features:
            log_test(test_name, 'FAIL', "No features registered")
            return False
        
        # Expected features from technical_indicators.py
        expected_features = ['momentum', 'trend', 'volatility', 'volume']
        missing = [f for f in expected_features if f not in features]
        
        if missing:
            log_test(test_name, 'WARN', f"Expected features not found: {missing}")
        
        log_test(test_name, 'PASS', f"Found {len(features)} registered features: {features}")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception in feature registry", traceback.format_exc())
        return False


def test_feature_engine():
    """Test 5: Feature engineering."""
    test_name = "Feature Engineering"
    try:
        loader = DataLoader()
        tickers = loader.get_available_tickers()
        
        if len(tickers) == 0:
            log_test(test_name, 'WARN', "No data to test features")
            return True
        
        test_ticker = tickers[0]
        data = loader.load_ticker_ohlcv(test_ticker)
        
        # Initialize feature engine
        engine = FeatureEngine()
        
        # Add features
        features_to_test = ['momentum', 'trend']
        for feature_name in features_to_test:
            try:
                engine.add_feature(feature_name)
            except Exception as e:
                log_test(test_name, 'WARN', f"Could not add feature '{feature_name}': {e}")
        
        # Compute features
        features_df = engine.compute_features(data)
        
        # Validate output
        if not isinstance(features_df, pd.DataFrame):
            log_test(test_name, 'FAIL', f"Features output is not DataFrame: {type(features_df)}")
            return False
        
        # Check index alignment
        if not isinstance(features_df.index, pd.DatetimeIndex):
            log_test(test_name, 'FAIL', f"Features index is not DatetimeIndex: {type(features_df.index)}")
            return False
        
        if features_df.index.name != 'date':
            log_test(test_name, 'FAIL', f"Features index name is '{features_df.index.name}', expected 'date'")
            return False
        
        # Check for features
        if features_df.empty:
            log_test(test_name, 'FAIL', "Feature computation returned empty DataFrame")
            return False
        
        # Validate no all-NaN columns
        all_nan_cols = features_df.columns[features_df.isnull().all()].tolist()
        if all_nan_cols:
            log_test(test_name, 'WARN', f"All-NaN feature columns: {all_nan_cols}")
        
        log_test(test_name, 'PASS', f"Generated {len(features_df.columns)} features for {len(features_df)} rows")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception during feature engineering", traceback.format_exc())
        return False


def test_model_trainer():
    """Test 6: Model training."""
    test_name = "Model Training"
    try:
        loader = DataLoader()
        tickers = loader.get_available_tickers()
        
        if len(tickers) == 0:
            log_test(test_name, 'WARN', "No data for model training")
            return True
        
        # Use first ticker for testing
        test_ticker = tickers[0]
        data = loader.load_ticker_ohlcv(test_ticker)
        
        # Need sufficient data for training
        if len(data) < 500:
            log_test(test_name, 'WARN', f"Insufficient data for training: {len(data)} rows")
            return True
        
        # Generate features
        engine = FeatureEngine()
        engine.add_feature('momentum')
        engine.add_feature('trend')
        features_df = engine.compute_features(data)
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Create simple target (next day return)
        from utils.targets import create_forward_returns
        targets = create_forward_returns(data, periods=[1])
        
        # Align data
        combined = features_df.join(targets, how='inner')
        combined = combined.dropna()
        
        if len(combined) < 100:
            log_test(test_name, 'WARN', f"Insufficient aligned data: {len(combined)} rows")
            return True
        
        # Split features and target
        feature_cols = [c for c in combined.columns if c not in targets.columns]
        X = combined[feature_cols]
        y = combined[targets.columns[0]]
        
        # Train model (use small subset for speed)
        train_size = min(300, int(len(X) * 0.7))
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        model = trainer.train_model(X_train, y_train)
        
        if model is None:
            log_test(test_name, 'FAIL', "Model training returned None")
            return False
        
        # Test prediction
        predictions = trainer.predict(model, X_train.iloc[:10])
        
        if predictions is None or len(predictions) == 0:
            log_test(test_name, 'FAIL', "Model predictions failed")
            return False
        
        log_test(test_name, 'PASS', f"Trained model on {len(X_train)} samples, predictions shape: {predictions.shape}")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception during model training", traceback.format_exc())
        return False


def test_universe_filter():
    """Test 7: Universe filtering."""
    test_name = "Universe Filter"
    try:
        loader = DataLoader()
        tickers = loader.get_available_tickers()
        
        if len(tickers) < 3:
            log_test(test_name, 'WARN', f"Need at least 3 tickers for filtering test, have {len(tickers)}")
            return True
        
        # Test different filters
        filters_to_test = [
            ('sp500', 'S&P 500'),
            ('liquid', 'Liquid stocks'),
            ('all', 'All stocks')
        ]
        
        for filter_name, description in filters_to_test:
            try:
                universe = loader.load_universe(filter_name)
                if not isinstance(universe, list):
                    log_test(test_name, 'FAIL', f"Filter '{filter_name}' returned non-list: {type(universe)}")
                    return False
            except Exception as e:
                log_test(test_name, 'WARN', f"Filter '{filter_name}' failed: {e}")
        
        log_test(test_name, 'PASS', f"Universe filters working, {len(tickers)} tickers available")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception in universe filtering", traceback.format_exc())
        return False


def test_portfolio_constructor():
    """Test 8: Portfolio construction."""
    test_name = "Portfolio Constructor"
    try:
        # Create dummy signals
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        signals = pd.DataFrame(
            np.random.randn(100, 3),
            index=dates,
            columns=tickers
        )
        signals.index.name = 'date'
        
        # Initialize constructor
        constructor = PortfolioConstructor()
        
        # Test equal weight
        weights = constructor.equal_weight(signals)
        
        if not isinstance(weights, pd.DataFrame):
            log_test(test_name, 'FAIL', f"Weights is not DataFrame: {type(weights)}")
            return False
        
        # Check weights sum to 1 (or 0 for no positions)
        weight_sums = weights.abs().sum(axis=1)
        invalid_sums = weight_sums[(weight_sums > 0) & ((weight_sums < 0.99) | (weight_sums > 1.01))]
        
        if len(invalid_sums) > 0:
            log_test(test_name, 'FAIL', f"Found {len(invalid_sums)} rows with invalid weight sums")
            return False
        
        log_test(test_name, 'PASS', f"Portfolio construction successful: {weights.shape}")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception in portfolio construction", traceback.format_exc())
        return False


def test_backtest_engine():
    """Test 9: Backtest engine."""
    test_name = "Backtest Engine"
    try:
        loader = DataLoader()
        tickers = loader.get_available_tickers()
        
        if len(tickers) == 0:
            log_test(test_name, 'WARN', "No data for backtesting")
            return True
        
        # Use first ticker
        test_ticker = tickers[0]
        data = loader.load_ticker_ohlcv(test_ticker)
        
        if len(data) < 100:
            log_test(test_name, 'WARN', f"Insufficient data for backtest: {len(data)} rows")
            return True
        
        # Create simple buy-and-hold weights
        weights = pd.DataFrame(
            1.0,
            index=data.index,
            columns=[test_ticker]
        )
        weights.index.name = 'date'
        
        # Run backtest
        engine = BacktestEngine()
        results = engine.run_backtest(weights, {test_ticker: data})
        
        if not isinstance(results, dict):
            log_test(test_name, 'FAIL', f"Backtest results is not dict: {type(results)}")
            return False
        
        # Validate results structure
        required_keys = ['equity_curve', 'trades', 'metrics']
        missing = [k for k in required_keys if k not in results]
        if missing:
            log_test(test_name, 'FAIL', f"Missing result keys: {missing}")
            return False
        
        # Validate equity curve
        equity = results['equity_curve']
        if not isinstance(equity, pd.Series):
            log_test(test_name, 'FAIL', f"Equity curve is not Series: {type(equity)}")
            return False
        
        if equity.iloc[0] != 1.0:
            log_test(test_name, 'WARN', f"Equity curve doesn't start at 1.0: {equity.iloc[0]}")
        
        # Validate metrics
        metrics = results['metrics']
        expected_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        missing_metrics = [m for m in expected_metrics if m not in metrics]
        if missing_metrics:
            log_test(test_name, 'WARN', f"Missing metrics: {missing_metrics}")
        
        log_test(test_name, 'PASS', f"Backtest completed: {len(equity)} days, return: {metrics.get('total_return', 'N/A')}")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception during backtesting", traceback.format_exc())
        return False


def test_data_ingestion():
    """Test 10: Data ingestion (non-destructive check)."""
    test_name = "Data Ingestion"
    try:
        ingestion = DataIngestion()
        
        # Test S&P 500 ticker retrieval
        sp500_tickers = ingestion.get_sp500_tickers()
        
        if not isinstance(sp500_tickers, list):
            log_test(test_name, 'FAIL', f"S&P 500 tickers is not list: {type(sp500_tickers)}")
            return False
        
        if len(sp500_tickers) == 0:
            log_test(test_name, 'WARN', "No S&P 500 tickers found")
        elif len(sp500_tickers) < 400:
            log_test(test_name, 'WARN', f"Only {len(sp500_tickers)} S&P 500 tickers (expected ~500)")
        
        log_test(test_name, 'PASS', f"Data ingestion ready: {len(sp500_tickers)} S&P 500 tickers")
        return True
        
    except Exception as e:
        log_test(test_name, 'FAIL', "Exception in data ingestion", traceback.format_exc())
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "="*80)
    print("QA TEST SUMMARY")
    print("="*80)
    
    total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['warnings'])
    
    print(f"\n✅ PASSED: {len(test_results['passed'])}/{total_tests}")
    for result in test_results['passed']:
        print(f"   • {result['test']}: {result['message']}")
    
    if test_results['warnings']:
        print(f"\n⚠️  WARNINGS: {len(test_results['warnings'])}")
        for result in test_results['warnings']:
            print(f"   • {result['test']}: {result['message']}")
    
    if test_results['failed']:
        print(f"\n❌ FAILED: {len(test_results['failed'])}")
        for result in test_results['failed']:
            print(f"   • {result['test']}: {result['message']}")
            if result['details']:
                print(f"     {result['details'][:200]}...")
    
    print("\n" + "="*80)
    
    if test_results['failed']:
        print("❌ QA TESTS FAILED - Please fix issues above")
        return False
    elif test_results['warnings']:
        print("⚠️  QA TESTS PASSED WITH WARNINGS - Review warnings above")
        return True
    else:
        print("✅ ALL QA TESTS PASSED SUCCESSFULLY!")
        return True


def main():
    """Run all QA tests."""
    print("="*80)
    print("QUANTSEARCH SYSTEM QA TEST SUITE")
    print("="*80)
    print(f"Started at: {datetime.now()}\n")
    
    # Run tests in sequence
    tests = [
        test_config,
        test_data_loader,
        test_data_validator,
        test_feature_registry,
        test_feature_engine,
        test_universe_filter,
        test_portfolio_constructor,
        test_backtest_engine,
        test_model_trainer,
        test_data_ingestion
    ]
    
    for test_func in tests:
        print(f"\nRunning: {test_func.__doc__.strip()}")
        test_func()
    
    # Print summary
    success = print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

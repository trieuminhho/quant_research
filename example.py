"""
Example: Complete Workflow
Demonstrates the full QuantSearch pipeline from data to backtest.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config import get_config
from utils.data_loader import DataLoader
from data.ingestion import DataIngestion
from features import FeatureEngine
from models.trainer import ModelTrainer
from strategies.base_strategy import MLStrategy
from backtest.engine import BacktestEngine
from backtest.portfolio import PortfolioConstructor


def example_1_download_data():
    """Example 1: Download S&P 500 data."""
    print("\n" + "="*60)
    print("Example 1: Download Historical Data")
    print("="*60)
    
    # Initialize data ingestion
    ingestion = DataIngestion(
        output_dir="data/raw/ohlcv",
        start_date="2015-01-01"
    )
    
    # Download a small sample (5 stocks for testing)
    sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print(f"\nDownloading {len(sample_tickers)} stocks...")
    results = ingestion.download_all_tickers(sample_tickers, max_workers=2)
    
    successful = sum(1 for v in results.values() if v)
    print(f"✅ Successfully downloaded {successful}/{len(sample_tickers)} stocks")
    
    # Validate data
    print("\nValidating data quality...")
    validation = ingestion.validate_data()
    valid_count = validation['is_valid'].sum()
    print(f"✅ {valid_count} stocks passed validation")


def example_2_engineer_features():
    """Example 2: Engineer features from OHLCV data."""
    print("\n" + "="*60)
    print("Example 2: Feature Engineering")
    print("="*60)
    
    # Load data
    loader = DataLoader()
    tickers = loader.get_available_tickers()[:5]  # First 5 stocks
    
    if not tickers:
        print("❌ No data available. Run example_1_download_data() first.")
        return
    
    print(f"\nLoading data for {len(tickers)} stocks...")
    data_dict = loader.load_multiple_tickers(tickers)
    
    # Create feature engine
    engine = FeatureEngine()
    
    # Add features
    print("\nAdding features:")
    features_to_add = ['returns', 'sma', 'ema', 'rsi', 'macd', 'volatility']
    for feature in features_to_add:
        engine.add_feature(feature)
        print(f"  ✓ {feature}")
    
    # Compute features
    print(f"\nComputing features for {len(data_dict)} stocks...")
    
    config = get_config()
    save_dir = config.get_path('processed_features')
    
    results = engine.compute_features_batch(data_dict, save_dir=str(save_dir))
    
    print(f"✅ Features computed and saved!")
    
    # Show sample
    sample_ticker = tickers[0]
    sample_features = results[sample_ticker]
    print(f"\nSample features for {sample_ticker}:")
    print(f"  Shape: {sample_features.shape}")
    print(f"  Columns: {list(sample_features.columns[:10])}...")
    print(f"\nSaved to: {save_dir}")


def example_3_portfolio_construction():
    """Example 3: Portfolio construction with signals."""
    print("\n" + "="*60)
    print("Example 3: Portfolio Construction")
    print("="*60)
    
    # Create sample signals
    signals = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
        'signal': [1, 1, 1, 0, 1, 1, -1],
        'predicted_return': [0.05, 0.08, 0.06, 0.02, 0.10, 0.07, -0.03],
        'volatility': [0.20, 0.18, 0.22, 0.25, 0.35, 0.30, 0.28],
        'confidence': [0.8, 0.9, 0.85, 0.6, 0.7, 0.88, 0.75]
    })
    
    print("\nInput signals:")
    print(signals.to_string(index=False))
    
    # Create portfolio constructor
    constructor = PortfolioConstructor({
        'max_positions': 5,
        'max_position_weight': 0.25,
        'max_turnover': 0.50
    })
    
    # Construct portfolio
    print("\nConstructing portfolio...")
    portfolio = constructor.construct_portfolio(
        signals=signals,
        current_holdings={},
        ranking_method='risk_adjusted',
        weighting_method='risk_parity'
    )
    
    # Display results
    print("\n✅ Portfolio constructed!")
    print("\nTarget Weights:")
    for ticker, weight in sorted(portfolio.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ticker:6s}: {weight:6.2%}")
    
    # Portfolio stats
    stats = constructor.calculate_portfolio_stats(portfolio)
    print(f"\nPortfolio Statistics:")
    print(f"  Positions: {stats['n_positions']}")
    print(f"  Max Weight: {stats['max_weight']:.2%}")
    print(f"  Concentration (HHI): {stats['concentration']:.3f}")


def example_4_simple_backtest():
    """Example 4: Simple backtest demonstration."""
    print("\n" + "="*60)
    print("Example 4: Simple Backtest")
    print("="*60)
    
    print("\n⚠️  Note: This is a simplified demonstration.")
    print("Full backtest requires trained ML models and predictions.")
    
    # Load some data
    loader = DataLoader()
    tickers = loader.get_available_tickers()[:3]
    
    if not tickers:
        print("❌ No data available. Run example_1_download_data() first.")
        return
    
    print(f"\nLoading data for {tickers}...")
    market_data = loader.load_multiple_tickers(tickers)
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000,
        commission_pct=0.001,
        slippage_bps=5
    )
    
    print("\n📊 Backtest Configuration:")
    print(f"  Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"  Commission: {engine.commission_pct:.3%}")
    print(f"  Slippage: {engine.slippage_bps*10000:.0f} bps")
    
    print("\n✅ Backtest engine initialized!")
    print("To run a full backtest:")
    print("  1. Train ML models (see example in README)")
    print("  2. Create strategy with predictions")
    print("  3. Call engine.run_backtest(strategy, market_data, ...)")


def example_5_streamlit_dashboard():
    """Example 5: Launch Streamlit dashboard."""
    print("\n" + "="*60)
    print("Example 5: Streamlit Dashboard")
    print("="*60)
    
    print("\nTo launch the interactive dashboard:")
    print("\n  streamlit run ui/dashboard.py")
    print("\nThis will start a web server at http://localhost:8501")
    print("\nThe dashboard provides:")
    print("  • Data management and download")
    print("  • Feature engineering interface")
    print("  • Model training controls")
    print("  • Backtesting and analysis")
    print("  • Performance visualization")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("QuantSearch Platform - Example Workflows")
    print("="*60)
    
    print("\nAvailable examples:")
    print("  1. Download historical data")
    print("  2. Engineer features")
    print("  3. Portfolio construction")
    print("  4. Simple backtest")
    print("  5. Streamlit dashboard")
    
    print("\n" + "="*60)
    
    # Run examples
    try:
        # Example 1: Download data
        # Uncomment to run:
        # example_1_download_data()
        
        # Example 2: Feature engineering
        # Uncomment to run (requires data from example 1):
        # example_2_engineer_features()
        
        # Example 3: Portfolio construction
        example_3_portfolio_construction()
        
        # Example 4: Backtest
        # Uncomment to run (requires data):
        # example_4_simple_backtest()
        
        # Example 5: Dashboard info
        example_5_streamlit_dashboard()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review README.md for detailed documentation")
    print("  2. Check docs/architecture.md for system design")
    print("  3. Customize config.yaml for your needs")
    print("  4. Start building your strategies!")


if __name__ == "__main__":
    main()

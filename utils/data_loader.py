"""
Data Loader Utilities
Provides functions to load and organize CSV data by asset, frequency, and data type.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and organizing data from CSV files.
    Provides efficient access to OHLCV, features, and prediction data.
    """
    
    def __init__(self, data_root: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_root: Root directory containing data folders
        """
        self.data_root = Path(data_root)
        self.raw_data_dir = self.data_root / "raw" / "ohlcv"
        self.features_dir = self.data_root / "processed" / "features"
        self.predictions_dir = self.data_root / "processed" / "predictions"
        
    def load_ticker_ohlcv(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with OHLCV data (date-indexed), or None if not found
        """
        filepath = self.raw_data_dir / f"{ticker}.csv"
        
        if not filepath.exists():
            logger.warning(f"OHLCV data not found for {ticker}")
            return None
        
        try:
            # Use parse_dates for efficiency, set index in one operation
            data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            
            # Ensure index is proper DatetimeIndex (not datetime.date objects)
            # This is critical for Plotly and other libraries that expect pd.Timestamp
            data.index = pd.DatetimeIndex(data.index)
            
            return data
        except Exception as e:
            logger.error(f"Error loading {ticker}: {e}")
            return None
    
    def load_multiple_tickers(self, 
                             tickers: List[str],
                             combine: bool = False) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load OHLCV data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            combine: If True, combine into single DataFrame with ticker column
            
        Returns:
            Dictionary of DataFrames (if combine=False) or single DataFrame (if combine=True)
        """
        data_dict = {}
        
        for ticker in tickers:
            data = self.load_ticker_ohlcv(ticker)
            if data is not None:
                data_dict[ticker] = data
        
        if combine:
            # Combine all tickers into single DataFrame
            if not data_dict:
                return pd.DataFrame()
            
            # More efficient concatenation
            dfs = []
            for ticker, df in data_dict.items():
                df_copy = df.copy()
                df_copy['ticker'] = ticker
                df_copy.reset_index(inplace=True)
                dfs.append(df_copy)
            
            combined = pd.concat(dfs, ignore_index=True)
            combined.sort_values(['ticker', 'date'], inplace=True)
            return combined
        
        return data_dict
    
    def load_all_tickers(self, combine: bool = False) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load OHLCV data for all available tickers.
        
        Args:
            combine: If True, combine into single DataFrame
            
        Returns:
            Dictionary of DataFrames or single DataFrame
        """
        # Get all CSV files in raw data directory
        csv_files = list(self.raw_data_dir.glob('*.csv'))
        csv_files = [f for f in csv_files if f.stem not in ['sp500_tickers', 'download_summary', 'data_validation_report']]
        
        tickers = [f.stem for f in csv_files]
        logger.info(f"Loading {len(tickers)} tickers...")
        
        return self.load_multiple_tickers(tickers, combine=combine)
    
    def get_available_tickers(self) -> List[str]:
        """
        Get list of all available tickers.
        
        Returns:
            List of ticker symbols
        """
        csv_files = list(self.raw_data_dir.glob('*.csv'))
        csv_files = [f for f in csv_files if f.stem not in ['sp500_tickers', 'download_summary', 'data_validation_report']]
        return sorted([f.stem for f in csv_files])
    
    def load_features(self, ticker: str, feature_set: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load engineered features for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            feature_set: Optional feature set name (e.g., 'technical', 'all')
            
        Returns:
            DataFrame with features (date-indexed), or None if not found
        """
        filename = f"{ticker}_{feature_set}_features.csv" if feature_set else f"{ticker}_features.csv"
        filepath = self.features_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Features not found for {ticker} ({feature_set})")
            return None
        
        try:
            # Use parse_dates and index_col for efficiency
            data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            
            # Ensure index is proper DatetimeIndex
            data.index = pd.DatetimeIndex(data.index)
            
            return data
        except Exception as e:
            logger.error(f"Error loading features for {ticker}: {e}")
            return None
    
    def load_predictions(self, ticker: str, model_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load model predictions for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            model_name: Optional model name
            
        Returns:
            DataFrame with predictions (date-indexed), or None if not found
        """
        filename = f"{ticker}_{model_name}_predictions.csv" if model_name else f"{ticker}_predictions.csv"
        filepath = self.predictions_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Predictions not found for {ticker} ({model_name})")
            return None
        
        try:
            # Use parse_dates and index_col for efficiency
            data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            
            # Ensure index is proper DatetimeIndex
            data.index = pd.DatetimeIndex(data.index)
            
            return data
        except Exception as e:
            logger.error(f"Error loading predictions for {ticker}: {e}")
            return None
    
    def get_date_range(self, tickers: Optional[List[str]] = None) -> tuple:
        """
        Get the common date range across tickers.
        
        Args:
            tickers: Optional list of tickers. If None, uses all available tickers.
            
        Returns:
            Tuple of (min_date, max_date)
        """
        if tickers is None:
            tickers = self.get_available_tickers()
        
        min_dates = []
        max_dates = []
        
        for ticker in tickers:
            data = self.load_ticker_ohlcv(ticker)
            if data is not None:
                min_dates.append(data.index.min())
                max_dates.append(data.index.max())
        
        if not min_dates:
            return None, None
        
        return max(min_dates), min(max_dates)
    
    def filter_by_date_range(self, 
                            data: pd.DataFrame,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            data: DataFrame with date index or date column
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Filtered DataFrame
        """
        df = data.copy()
        
        # Ensure date index
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('date', inplace=True)
        
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Apply filters using boolean indexing (more efficient)
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        return df
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for all available data.
        
        Returns:
            DataFrame with summary statistics per ticker
        """
        tickers = self.get_available_tickers()
        summary_data = []
        
        for ticker in tickers:
            data = self.load_ticker_ohlcv(ticker)
            if data is not None:
                summary_data.append({
                    'ticker': ticker,
                    'num_records': len(data),
                    'start_date': data.index.min(),
                    'end_date': data.index.max(),
                    'avg_volume': data['volume'].mean() if 'volume' in data.columns else 0,
                    'avg_close': data['close'].mean() if 'close' in data.columns else 0
                })
        
        return pd.DataFrame(summary_data)


class UniverseFilter:
    """
    Filters stock universe based on data quality and availability criteria.
    """
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize universe filter.
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
    
    def filter_by_data_quality(self,
                              min_data_points: int = 252,
                              max_missing_pct: float = 0.10) -> List[str]:
        """
        Filter tickers by data quality criteria.
        
        Args:
            min_data_points: Minimum number of data points required
            max_missing_pct: Maximum percentage of missing data allowed
            
        Returns:
            List of valid tickers
        """
        all_tickers = self.data_loader.get_available_tickers()
        valid_tickers = []
        
        for ticker in all_tickers:
            data = self.data_loader.load_ticker_ohlcv(ticker)
            
            if data is None:
                continue
            
            # Check number of data points
            if len(data) < min_data_points:
                continue
            
            # Check missing data
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_pct > max_missing_pct:
                continue
            
            valid_tickers.append(ticker)
        
        logger.info(f"Filtered universe: {len(valid_tickers)}/{len(all_tickers)} tickers pass quality criteria")
        return valid_tickers
    
    def filter_by_date_availability(self,
                                   start_date: str,
                                   end_date: str,
                                   min_coverage: float = 0.95) -> List[str]:
        """
        Filter tickers by date range coverage.
        
        Args:
            start_date: Required start date
            end_date: Required end date
            min_coverage: Minimum percentage of dates that must be present
            
        Returns:
            List of tickers with sufficient coverage
        """
        all_tickers = self.data_loader.get_available_tickers()
        valid_tickers = []
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        required_days = len(pd.bdate_range(start_dt, end_dt))
        
        for ticker in all_tickers:
            data = self.data_loader.load_ticker_ohlcv(ticker)
            
            if data is None:
                continue
            
            # Filter to date range
            data_in_range = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            # Check coverage
            coverage = len(data_in_range) / required_days
            if coverage >= min_coverage:
                valid_tickers.append(ticker)
        
        logger.info(f"Date filter: {len(valid_tickers)}/{len(all_tickers)} tickers have {min_coverage*100}% coverage")
        return valid_tickers


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Get available tickers
    tickers = loader.get_available_tickers()
    print(f"Available tickers: {len(tickers)}")
    
    # Load single ticker
    if tickers:
        data = loader.load_ticker_ohlcv(tickers[0])
        print(f"\n{tickers[0]} data shape: {data.shape}")
        print(data.head())
    
    # Get data summary
    summary = loader.get_data_summary()
    print(f"\nData summary:\n{summary.head()}")
    
    # Filter universe
    filter_obj = UniverseFilter(loader)
    valid_tickers = filter_obj.filter_by_data_quality(min_data_points=252)
    print(f"\nValid tickers: {len(valid_tickers)}")

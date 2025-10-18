"""
Data Ingestion Module
Downloads and stores historical market data for S&P 500 stocks in CSV format.
"""

import os
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles downloading and storing market data for S&P 500 stocks.
    All data is stored locally in CSV format with organized folder structure.
    Supports multiple timeframes: 15m, 30m, 1h, 4h, 12h, 1d, 1w, 1M
    """
    
    # Mapping of our timeframe notation to yfinance intervals
    TIMEFRAME_MAP = {
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '12h': '1d',  # yfinance doesn't have 12h, will need custom logic
        '1d': '1d',
        '1w': '1wk',
        '1M': '1mo'
    }
    
    # Maximum history available for each timeframe (yfinance limitations)
    TIMEFRAME_LIMITS = {
        '15m': 60,    # days
        '30m': 60,    # days
        '1h': 730,    # days (~2 years)
        '4h': 730,    # days
        '12h': 730,   # days
        '1d': 36500,  # days (no practical limit)
        '1w': 36500,  # days
        '1M': 36500   # days
    }
    
    def __init__(self, 
                 output_dir: str = "data/raw/ohlcv",
                 start_date: str = "2010-01-01",
                 end_date: Optional[str] = None,
                 timeframes: Optional[List[str]] = None):
        """
        Initialize data ingestion.
        
        Args:
            output_dir: Base directory to save CSV files
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD). None = today
            timeframes: List of timeframes to download (e.g., ['1d', '1h', '15m'])
                       If None, defaults to ['1d']
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        
        # Default to daily if not specified
        self.timeframes = timeframes if timeframes else ['1d']
        
        # Create subdirectories for each timeframe
        for tf in self.timeframes:
            tf_dir = self.output_dir / tf
            tf_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataIngestion initialized: {self.start_date} to {self.end_date} (timeframes: {self.timeframes})")
    
    def get_sp500_tickers(self) -> List[str]:
        """
        Get current S&P 500 constituent tickers from Wikipedia.
        
        Returns:
            List of ticker symbols
        """
        logger.info("Fetching S&P 500 tickers from Wikipedia...")
        
        try:
            # Fetch from Wikipedia with headers to avoid 403 error
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            
            # Add User-Agent header to avoid blocking
            import urllib.request
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req) as response:
                tables = pd.read_html(response.read())
            
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean tickers (replace dots with dashes for Yahoo Finance)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            logger.info(f"Found {len(tickers)} S&P 500 tickers")
            
            # Save tickers list
            tickers_df = pd.DataFrame({'ticker': tickers})
            tickers_path = self.output_dir / 'sp500_tickers.csv'
            tickers_df.to_csv(tickers_path, index=False)
            logger.info(f"Saved tickers to {tickers_path}")
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            # Fallback to a small list for testing
            logger.warning("Using fallback ticker list")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    def download_ticker_data(self, 
                            ticker: str,
                            timeframe: str = '1d',
                            retries: int = 3,
                            delay: float = 1.0) -> Optional[pd.DataFrame]:
        """
        Download historical OHLCV data for a single ticker at specified timeframe.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe (15m, 30m, 1h, 4h, 12h, 1d, 1w, 1M)
            retries: Number of retry attempts
            delay: Delay between retries (seconds)
            
        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        # Validate timeframe
        if timeframe not in self.TIMEFRAME_MAP:
            logger.error(f"Invalid timeframe: {timeframe}. Valid options: {list(self.TIMEFRAME_MAP.keys())}")
            return None
        
        # Adjust start date based on timeframe limitations
        start_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
        max_days = self.TIMEFRAME_LIMITS.get(timeframe, 36500)
        
        # Calculate the actual start date respecting limitations
        earliest_start = end_date_obj - timedelta(days=max_days)
        actual_start = max(start_date_obj, earliest_start)
        
        for attempt in range(retries):
            try:
                logger.info(f"Downloading {ticker} [{timeframe}] (attempt {attempt + 1}/{retries})...")
                
                # Get yfinance interval
                yf_interval = self.TIMEFRAME_MAP[timeframe]
                
                # For intraday data (15m, 30m, 1h, 4h), yfinance requires 'period' parameter
                # instead of start/end dates
                if timeframe in ['15m', '30m', '1h', '4h']:
                    # Calculate period in days
                    days_back = min(max_days, 60)  # Max 60 days for intraday
                    
                    # Use period parameter for intraday data
                    data = yf.download(
                        ticker,
                        period=f"{days_back}d",
                        interval=yf_interval,
                        progress=False,
                        auto_adjust=False
                    )
                elif timeframe == '12h':
                    # For 12h, download 1d and keep as is (limitation)
                    data = yf.download(
                        ticker,
                        start=actual_start.strftime('%Y-%m-%d'),
                        end=self.end_date,
                        interval='1d',
                        progress=False,
                        auto_adjust=False
                    )
                else:
                    # For daily and higher timeframes, use start/end dates
                    data = yf.download(
                        ticker,
                        start=actual_start.strftime('%Y-%m-%d'),
                        end=self.end_date,
                        interval=yf_interval,
                        progress=False,
                        auto_adjust=False
                    )
                
                if data.empty:
                    logger.warning(f"No data returned for {ticker} [{timeframe}]")
                    return None
                
                # Flatten multi-level columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Standardize column names (lowercase with underscores)
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                
                # Reset index to make date a column
                data.reset_index(inplace=True)
                
                # Standardize date column name
                if 'Date' in data.columns:
                    data.rename(columns={'Date': 'date'}, inplace=True)
                elif 'Datetime' in data.columns:
                    data.rename(columns={'Datetime': 'date'}, inplace=True)
                elif data.index.name in ['Date', 'Datetime']:
                    data.index.name = 'date'
                    data.reset_index(inplace=True)
                
                # Ensure date is datetime
                data['date'] = pd.to_datetime(data['date'])
                
                # Add ticker and timeframe columns
                data['ticker'] = ticker
                data['timeframe'] = timeframe
                
                # Sort by date
                data.sort_values('date', inplace=True)
                
                logger.info(f"Successfully downloaded {len(data)} rows for {ticker} [{timeframe}]")
                return data
                
            except Exception as e:
                logger.warning(f"Error downloading {ticker} [{timeframe}] (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to download {ticker} [{timeframe}] after {retries} attempts")
                    return None
        
        return None
    
    def save_ticker_data(self, ticker: str, data: pd.DataFrame, timeframe: str = '1d'):
        """
        Save ticker data to CSV file in timeframe-specific directory.
        
        Args:
            ticker: Stock ticker symbol
            data: DataFrame with OHLCV data
            timeframe: Timeframe (15m, 30m, 1h, 4h, 12h, 1d, 1w, 1M)
        """
        # Create timeframe directory
        tf_dir = self.output_dir / timeframe
        tf_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{ticker}.csv"
        filepath = tf_dir / filename
        
        # Ensure date column is string format for consistent CSV storage
        data_to_save = data.copy()
        if 'date' in data_to_save.columns:
            data_to_save['date'] = pd.to_datetime(data_to_save['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to CSV
        data_to_save.to_csv(filepath, index=False)
        logger.info(f"Saved {ticker} [{timeframe}] data to {filepath}")
    
    def download_all_tickers(self, 
                            tickers: Optional[List[str]] = None,
                            timeframes: Optional[List[str]] = None,
                            max_workers: int = 5,
                            batch_delay: float = 1.0,
                            skip_existing: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Download data for all tickers across multiple timeframes (parallelized with rate limiting).
        
        Args:
            tickers: List of tickers. If None, fetches S&P 500 tickers
            timeframes: List of timeframes to download. If None, uses instance timeframes
            max_workers: Maximum number of parallel downloads
            batch_delay: Delay between batches to avoid rate limiting (seconds)
            skip_existing: Skip downloading if file already exists
            
        Returns:
            Nested dictionary: {timeframe: {ticker: 'downloaded'|'skipped'|'failed'}}
        """
        if tickers is None:
            tickers = self.get_sp500_tickers()
        
        if timeframes is None:
            timeframes = self.timeframes
        
        logger.info(f"Starting download for {len(tickers)} tickers across {len(timeframes)} timeframe(s)...")
        
        results = {}
        
        # Process each timeframe
        for tf in timeframes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing timeframe: {tf}")
            logger.info(f"{'='*60}")
            
            results[tf] = {}
            
            # Check which tickers need downloading
            tickers_to_download = []
            for ticker in tickers:
                tf_dir = self.output_dir / tf
                filepath = tf_dir / f"{ticker}.csv"
                
                if skip_existing and filepath.exists():
                    logger.info(f"Skipping {ticker} [{tf}] - already exists")
                    results[tf][ticker] = 'skipped'
                else:
                    tickers_to_download.append(ticker)
            
            if not tickers_to_download:
                logger.info(f"All tickers already downloaded for {tf}")
                continue
            
            logger.info(f"Downloading {len(tickers_to_download)} tickers for {tf}...")
            
            # Process in batches to avoid overwhelming the API
            batch_size = max_workers
            for i in range(0, len(tickers_to_download), batch_size):
                batch = tickers_to_download[i:i + batch_size]
                
                logger.info(f"Processing batch {i // batch_size + 1}/{(len(tickers_to_download) + batch_size - 1) // batch_size}: {len(batch)} tickers")
                
                # Download batch in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ticker = {
                        executor.submit(self.download_ticker_data, ticker, tf): ticker 
                        for ticker in batch
                    }
                    
                    for future in as_completed(future_to_ticker):
                        ticker = future_to_ticker[future]
                        try:
                            data = future.result()
                            if data is not None:
                                self.save_ticker_data(ticker, data, tf)
                                results[tf][ticker] = 'downloaded'
                            else:
                                results[tf][ticker] = 'failed'
                        except Exception as e:
                            logger.error(f"Unexpected error processing {ticker} [{tf}]: {e}")
                            results[tf][ticker] = 'failed'
                
                # Rate limiting delay between batches (more aggressive for intraday data)
                if i + batch_size < len(tickers_to_download):
                    delay = batch_delay * 2 if tf in ['15m', '30m', '1h'] else batch_delay
                    logger.info(f"Rate limiting: waiting {delay}s before next batch...")
                    time.sleep(delay)
        
        # Summary
        for tf, tf_results in results.items():
            successful = sum(1 for v in tf_results.values() if v == 'downloaded')
            skipped = sum(1 for v in tf_results.values() if v == 'skipped')
            failed = sum(1 for v in tf_results.values() if v == 'failed')
            logger.info(f"{tf}: {successful} downloaded, {skipped} skipped, {failed} failed")
        
        # Save summary
        summary_data = []
        for tf, tf_results in results.items():
            for ticker, status in tf_results.items():
                summary_data.append({
                    'ticker': ticker,
                    'timeframe': tf,
                    'status': status
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'download_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved download summary to {summary_path}")
        
        return results
    
    def update_existing_data(self, tickers: Optional[List[str]] = None):
        """
        Update existing CSV files with new data (incremental update).
        
        Args:
            tickers: List of tickers to update. If None, updates all existing files
        """
        if tickers is None:
            # Get all existing CSV files
            csv_files = list(self.output_dir.glob('*.csv'))
            csv_files = [f for f in csv_files if f.stem not in ['sp500_tickers', 'download_summary']]
            tickers = [f.stem for f in csv_files]
        
        logger.info(f"Updating {len(tickers)} existing tickers...")
        
        for ticker in tickers:
            filepath = self.output_dir / f"{ticker}.csv"
            
            if not filepath.exists():
                logger.warning(f"File not found for {ticker}, downloading from scratch")
                data = self.download_ticker_data(ticker)
                if data is not None:
                    self.save_ticker_data(ticker, data)
                continue
            
            # Load existing data
            existing_data = pd.read_csv(filepath, parse_dates=['date'])
            
            # Get last date
            last_date = existing_data['date'].max()
            
            # Check if update is needed
            today = pd.Timestamp.now().normalize()
            if (today - last_date).days < 1:
                logger.info(f"{ticker} is up to date")
                continue
            
            # Download new data
            new_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info(f"Updating {ticker} from {new_start}...")
            
            try:
                new_data = yf.download(
                    ticker,
                    start=new_start,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=False
                )
                
                if not new_data.empty:
                    # Process new data (use same logic as download_ticker_data)
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = new_data.columns.get_level_values(0)
                    
                    new_data.columns = [col.lower().replace(' ', '_') for col in new_data.columns]
                    new_data.reset_index(inplace=True)
                    
                    if 'Date' in new_data.columns:
                        new_data.rename(columns={'Date': 'date'}, inplace=True)
                    elif new_data.index.name == 'Date':
                        new_data.index.name = 'date'
                        new_data.reset_index(inplace=True)
                    
                    new_data['date'] = pd.to_datetime(new_data['date'])
                    new_data['ticker'] = ticker
                    
                    # Concatenate and remove duplicates (keep='last' for most recent data)
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                    combined_data.drop_duplicates(subset=['date'], keep='last', inplace=True)
                    combined_data.sort_values('date', inplace=True)
                    
                    # Save updated data
                    self.save_ticker_data(ticker, combined_data)
                    logger.info(f"Updated {ticker} with {len(new_data)} new rows")
                else:
                    logger.info(f"No new data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error updating {ticker}: {e}")
    
    def validate_data(self, min_data_points: int = 252) -> pd.DataFrame:
        """
        Validate downloaded data and generate data quality report.
        
        Args:
            min_data_points: Minimum number of data points required
            
        Returns:
            DataFrame with validation results
        """
        logger.info("Validating downloaded data...")
        
        csv_files = list(self.output_dir.glob('*.csv'))
        csv_files = [f for f in csv_files if f.stem not in ['sp500_tickers', 'download_summary']]
        
        validation_results = []
        
        for filepath in csv_files:
            ticker = filepath.stem
            
            try:
                # Use parse_dates for efficiency
                data = pd.read_csv(filepath, parse_dates=['date'])
                
                # Calculate metrics (vectorized operations)
                num_rows = len(data)
                date_range = (data['date'].max() - data['date'].min()).days
                missing_values = data.isnull().sum().sum()
                missing_pct = missing_values / (len(data) * len(data.columns))
                
                # Check for price anomalies (vectorized)
                price_cols = ['open', 'high', 'low', 'close']
                available_price_cols = [col for col in price_cols if col in data.columns]
                
                has_zero_prices = (data[available_price_cols] == 0).any().any() if available_price_cols else False
                has_negative_prices = (data[available_price_cols] < 0).any().any() if available_price_cols else False
                
                is_valid = (
                    num_rows >= min_data_points and
                    not has_zero_prices and
                    not has_negative_prices and
                    missing_pct < 0.1
                )
                
                validation_results.append({
                    'ticker': ticker,
                    'num_rows': num_rows,
                    'date_range_days': date_range,
                    'missing_pct': missing_pct,
                    'has_zero_prices': has_zero_prices,
                    'has_negative_prices': has_negative_prices,
                    'is_valid': is_valid
                })
                
            except Exception as e:
                logger.error(f"Error validating {ticker}: {e}")
                validation_results.append({
                    'ticker': ticker,
                    'num_rows': 0,
                    'date_range_days': 0,
                    'missing_pct': 1.0,
                    'has_zero_prices': True,
                    'has_negative_prices': True,
                    'is_valid': False
                })
        
        validation_df = pd.DataFrame(validation_results)
        
        # Save validation report
        report_path = self.output_dir / 'data_validation_report.csv'
        validation_df.to_csv(report_path, index=False)
        logger.info(f"Saved validation report to {report_path}")
        
        # Summary
        valid_count = validation_df['is_valid'].sum()
        logger.info(f"Validation complete: {valid_count}/{len(validation_df)} tickers are valid")
        
        return validation_df


if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion(
        output_dir="data/raw/ohlcv",
        start_date="2010-01-01"
    )
    
    # Download all S&P 500 data
    results = ingestion.download_all_tickers(max_workers=5)
    
    # Validate data
    validation_report = ingestion.validate_data()
    
    print("\nData Ingestion Complete!")
    print(f"Total tickers processed: {len(results)}")
    print(f"Successful downloads: {sum(1 for v in results.values() if v)}")
    print(f"Valid tickers: {validation_report['is_valid'].sum()}")

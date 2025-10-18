"""
Data Validation Module - Multi-Source Version
Cross-validates stock data from multiple sources to ensure accuracy.
Supports yfinance, Alpha Vantage, Polygon.io, and Twelve Data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates stock data by cross-referencing multiple sources.
    Provides confidence scores and identifies discrepancies.
    Supports: yfinance, Alpha Vantage, Polygon.io, Twelve Data
    """
    
    def __init__(self, config=None):
        """Initialize data validator with optional config."""
        if config is None:
            from utils.config import get_config
            config = get_config()
        
        self.config = config
        
        # Load validation settings from config
        config_dict = config._config if hasattr(config, '_config') else {}
        data_config = config_dict.get('data', {})
        validation_config = data_config.get('validation', {})
        
        self.providers_config = validation_config.get('providers', {})
        self.min_providers = validation_config.get('min_providers', 2)
        self.consensus_threshold = validation_config.get('consensus_threshold', 0.01)
        self.default_tolerance = validation_config.get('tolerance', 0.02)
        self.lookback_days = validation_config.get('lookback_days', 10)
        
        # Initialize enabled providers
        self.enabled_providers = []
        for provider, settings in self.providers_config.items():
            if settings.get('enabled', False):
                self.enabled_providers.append(provider)
        
        logger.info(f"DataValidator initialized with providers: {', '.join(self.enabled_providers)}")
    
    def _fetch_alpha_vantage_data(self, ticker: str, days: int = 10) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage API."""
        try:
            api_key = self.providers_config.get('alpha_vantage', {}).get('api_key')
            if not api_key:
                logger.warning("Alpha Vantage API key not configured")
                return None
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data_json = response.json()
            
            if 'Error Message' in data_json:
                logger.error(f"Alpha Vantage error for {ticker}: {data_json['Error Message']}")
                return None
            
            if 'Note' in data_json:
                logger.warning(f"Alpha Vantage rate limit: {data_json['Note']}")
                return None
            
            time_series = data_json.get('Time Series (Daily)', {})
            if not time_series:
                return None
            
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.apply(pd.to_numeric)
            df = df.tail(days)
            
            logger.info(f"Alpha Vantage: Fetched {len(df)} days for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Alpha Vantage error for {ticker}: {e}")
            return None
    
    def _fetch_polygon_data(self, ticker: str, days: int = 10) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io API."""
        try:
            api_key = self.providers_config.get('polygon', {}).get('api_key')
            if not api_key:
                logger.warning("Polygon API key not configured")
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+5)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apiKey': api_key, 'adjusted': 'true', 'sort': 'asc'}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data_json = response.json()
            
            if data_json.get('status') == 'DELAYED':
                logger.warning(f"Polygon status: DELAYED for {ticker}")
            
            if data_json.get('status') != 'OK' and data_json.get('status') != 'DELAYED':
                logger.error(f"Polygon error for {ticker}: {data_json.get('status')}")
                return None
            
            results = data_json.get('results', [])
            if not results:
                return None
            
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('date', inplace=True)
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df[['open', 'high', 'low', 'close', 'volume']].tail(days)
            
            logger.info(f"Polygon: Fetched {len(df)} days for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Polygon error for {ticker}: {e}")
            return None
    
    def _fetch_finnhub_data(self, ticker: str, days: int = 10) -> Optional[pd.DataFrame]:
        """Fetch data from Finnhub API (60 calls/min free tier)."""
        try:
            api_key = self.providers_config.get('finnhub', {}).get('api_key')
            if not api_key:
                logger.warning("Finnhub API key not configured")
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+5)
            
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': ticker,
                'resolution': 'D',  # Daily
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp()),
                'token': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data_json = response.json()
            
            if data_json.get('s') == 'no_data':
                logger.warning(f"Finnhub: No data for {ticker}")
                return None
            
            if data_json.get('s') != 'ok':
                logger.error(f"Finnhub error for {ticker}: {data_json.get('s')}")
                return None
            
            # Build DataFrame
            df = pd.DataFrame({
                'open': data_json['o'],
                'high': data_json['h'],
                'low': data_json['l'],
                'close': data_json['c'],
                'volume': data_json['v']
            })
            df['date'] = pd.to_datetime(data_json['t'], unit='s')
            df.set_index('date', inplace=True)
            df = df.sort_index().tail(days)
            
            logger.info(f"Finnhub: Fetched {len(df)} days for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Finnhub error for {ticker}: {e}")
            return None
    
    def _fetch_twelve_data(self, ticker: str, days: int = 10) -> Optional[pd.DataFrame]:
        """Fetch data from Twelve Data API (800 calls/day free tier)."""
        try:
            api_key = self.providers_config.get('twelve_data', {}).get('api_key')
            if not api_key:
                logger.warning("Twelve Data API key not configured")
                return None
            
            # Twelve Data requires specific format
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': ticker,
                'interval': '1day',
                'outputsize': days + 5,  # Request extra to ensure we get enough data
                'apikey': api_key,
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data_json = response.json()
            
            # Check for errors
            if 'status' in data_json and data_json['status'] == 'error':
                logger.error(f"Twelve Data error for {ticker}: {data_json.get('message', 'Unknown error')}")
                return None
            
            if 'values' not in data_json or not data_json['values']:
                logger.warning(f"Twelve Data: No data for {ticker}")
                return None
            
            # Build DataFrame
            values = data_json['values']
            df = pd.DataFrame(values)
            
            # Convert and rename columns
            df['date'] = pd.to_datetime(df['datetime'])
            df = df.set_index('date')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.sort_index().tail(days)
            
            logger.info(f"Twelve Data: Fetched {len(df)} days for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Twelve Data error for {ticker}: {e}")
            return None
    
    def _fetch_polygon_data(self, ticker: str, days: int = 10) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io API (5 calls/min free tier)."""
        try:
            api_key = self.providers_config.get('polygon', {}).get('api_key')
            if not api_key:
                logger.warning("Polygon API key not configured")
                return None
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+5)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apiKey': api_key, 'adjusted': 'false'}
            
            time.sleep(0.2)  # Rate limit: 5 calls/sec
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data_json = response.json()
            
            if data_json.get('status') != 'OK':
                logger.warning(f"Polygon status: {data_json.get('status')} for {ticker}")
                return None
            
            results = data_json.get('results', [])
            if not results:
                return None
            
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('date')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.sort_index()
            
            logger.info(f"Polygon: Fetched {len(df)} days for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Polygon error for {ticker}: {e}")
            return None
    
    def _fetch_yfinance_data(self, ticker: str, days: int = 10) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+5)
            
            yf_data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False
            )
            
            if yf_data.empty:
                return None
            
            if isinstance(yf_data.columns, pd.MultiIndex):
                yf_data.columns = yf_data.columns.get_level_values(0)
            yf_data.columns = [col.lower().replace(' ', '_') for col in yf_data.columns]
            
            logger.info(f"yfinance: Fetched {len(yf_data)} days for {ticker}")
            return yf_data
            
        except Exception as e:
            logger.error(f"yfinance error for {ticker}: {e}")
            return None
        
    def validate_price_data(self, 
                           ticker: str, 
                           local_data: pd.DataFrame,
                           tolerance: float = None) -> Dict:
        """
        Validate local price data against multiple providers with consensus scoring.
        
        Args:
            ticker: Stock ticker symbol
            local_data: DataFrame with local OHLCV data
            tolerance: Acceptable difference threshold (uses config default if None)
            
        Returns:
            Dictionary with validation results including multi-source consensus
        """
        if tolerance is None:
            tolerance = self.default_tolerance
            
        results = {
            'ticker': ticker,
            'is_valid': False,
            'confidence_score': 0.0,
            'discrepancies': [],
            'last_validated': datetime.now().isoformat(),
            'data_quality': 'UNKNOWN',
            'providers_used': [],
            'consensus_score': 0.0
        }
        
        try:
            # Fetch from all enabled providers
            provider_data = {}
            
            for provider in self.enabled_providers:
                if provider == 'yfinance':
                    data = self._fetch_yfinance_data(ticker, self.lookback_days)
                elif provider == 'alpha_vantage':
                    data = self._fetch_alpha_vantage_data(ticker, self.lookback_days)
                elif provider == 'polygon':
                    data = self._fetch_polygon_data(ticker, self.lookback_days)
                elif provider == 'twelve_data':
                    data = self._fetch_twelve_data(ticker, self.lookback_days)
                elif provider == 'finnhub':
                    data = self._fetch_finnhub_data(ticker, self.lookback_days)
                else:
                    continue
                
                if data is not None and not data.empty:
                    provider_data[provider] = data
                    results['providers_used'].append(provider)
            
            if len(provider_data) == 0:
                results['discrepancies'].append("No data from any provider")
                results['data_quality'] = 'UNVERIFIED'
                return results
            
            # Prepare local data
            logger.info(f"{ticker} - Local data: {len(local_data)} rows, index type: {type(local_data.index).__name__}")
            local_data = local_data.copy()
            if not isinstance(local_data.index, pd.DatetimeIndex):
                logger.warning(f"{ticker} - Converting local data index to DatetimeIndex")
                if 'date' in local_data.columns:
                    local_data['date'] = pd.to_datetime(local_data['date'])
                    local_data.set_index('date', inplace=True)
                else:
                    results['discrepancies'].append("Local data missing date column")
                    results['data_quality'] = 'INVALID'
                    return results
            
            logger.info(f"{ticker} - Local data date range: {local_data.index.min()} to {local_data.index.max()}")
            
            # Compare against each provider
            provider_differences = {}
            
            for provider, provider_df in provider_data.items():
                common_dates = local_data.index.intersection(provider_df.index)
                logger.info(f"{ticker} - {provider}: {len(common_dates)} common dates out of "
                           f"{len(local_data)} local, {len(provider_df)} provider")
                
                if len(common_dates) == 0:
                    logger.warning(f"{ticker} - {provider}: No overlapping dates!")
                    logger.debug(f"  Local dates: {local_data.index.min()} to {local_data.index.max()}")
                    logger.debug(f"  Provider dates: {provider_df.index.min()} to {provider_df.index.max()}")
                    continue
                
                local_closes = local_data.loc[common_dates, 'close']
                provider_closes = provider_df.loc[common_dates, 'close']
                
                differences = []
                for date in common_dates[-5:]:
                    if date in local_closes.index and date in provider_closes.index:
                        local_price = local_closes.loc[date]
                        provider_price = provider_closes.loc[date]
                        diff_pct = abs((local_price - provider_price) / provider_price) if provider_price != 0 else 0
                        differences.append(diff_pct)
                        # Changed to INFO level so it always shows
                        logger.info(f"  {ticker} - {provider} - {date.strftime('%Y-%m-%d')}: "
                                  f"Local=${local_price:.2f}, Provider=${provider_price:.2f}, "
                                  f"diff={diff_pct*100:.2f}%")
                
                if differences:
                    avg_diff = np.mean(differences)
                    max_diff = max(differences)
                    provider_differences[provider] = {
                        'avg_diff': avg_diff,
                        'max_diff': max_diff,
                        'dates_checked': len(differences)
                    }
                    logger.info(f"{ticker} - {provider}: avg_diff={avg_diff*100:.3f}%, max_diff={max_diff*100:.3f}%")
            
            if not provider_differences:
                logger.warning(f"{ticker}: No overlapping dates with any provider")
                results['discrepancies'].append("No overlapping dates")
                results['data_quality'] = 'STALE'
                return results
            
            # Calculate consensus
            avg_diffs = [pd['avg_diff'] for pd in provider_differences.values()]
            overall_avg_diff = np.mean(avg_diffs)
            consensus_score = 100 - (np.std(avg_diffs) * 100 * 50)
            
            # Calculate confidence
            confidence = max(0, min(100, 100 - (overall_avg_diff * 100 * 20)))
            results['confidence_score'] = round(confidence, 2)
            results['consensus_score'] = round(max(0, consensus_score), 2)
            
            logger.info(f"{ticker} - Validation metrics: overall_avg_diff={overall_avg_diff*100:.4f}%, "
                       f"confidence={confidence:.2f}%, consensus={consensus_score:.2f}%")
            logger.info(f"{ticker} - Thresholds: EXCELLENT<0.1%, GOOD<0.5%, ACCEPTABLE<{tolerance*100}%")
            
            # Determine quality
            if overall_avg_diff < 0.001:
                results['data_quality'] = 'EXCELLENT'
                results['is_valid'] = True
                logger.info(f"{ticker}: Quality=EXCELLENT (diff {overall_avg_diff*100:.4f}% < 0.1%)")
            elif overall_avg_diff < 0.005:
                results['data_quality'] = 'GOOD'
                results['is_valid'] = True
                logger.info(f"{ticker}: Quality=GOOD (diff {overall_avg_diff*100:.4f}% < 0.5%)")
            elif overall_avg_diff < tolerance:
                results['data_quality'] = 'ACCEPTABLE'
                results['is_valid'] = True
                logger.info(f"{ticker}: Quality=ACCEPTABLE (diff {overall_avg_diff*100:.4f}% < {tolerance*100}%)")
            else:
                results['data_quality'] = 'POOR'
                results['is_valid'] = False
                logger.warning(f"{ticker}: Quality=POOR (diff {overall_avg_diff*100:.4f}% >= {tolerance*100}%)")
                
                # Add helpful diagnostic for large differences
                if overall_avg_diff > 0.3:  # More than 30% difference
                    logger.error(f"❌ {ticker}: CRITICAL DATA ISSUE - {overall_avg_diff*100:.1f}% price difference!")
                    logger.error(f"   This likely indicates corrupted data or stock split mismatch.")
                    logger.error(f"   SOLUTION: Re-download {ticker} data to fix the issue.")
                    results['discrepancies'].append(
                        f"CRITICAL: {overall_avg_diff*100:.1f}% price difference suggests corrupted data. "
                        f"Please re-download this ticker."
                    )
            
            results['avg_difference_pct'] = float(overall_avg_diff * 100)
            results['max_difference_pct'] = float(max([pd['max_diff'] for pd in provider_differences.values()]) * 100)
            results['dates_checked'] = max([pd['dates_checked'] for pd in provider_differences.values()])
            
            logger.info(f"{ticker}: {results['data_quality']} (confidence: {results['confidence_score']:.1f}%, "
                       f"consensus: {results['consensus_score']:.1f}%, providers: {len(provider_data)})")
            
        except Exception as e:
            logger.error(f"Validation error for {ticker}: {e}")
            results['discrepancies'].append(f"Error: {str(e)}")
            results['data_quality'] = 'ERROR'
        
        return results


    def validate_ticker_multisource(self, ticker: str, sources: List[str] = None) -> Dict:
        """
        Simplified validation interface that loads local data and validates against sources.
        
        Args:
            ticker: Stock ticker symbol
            sources: List of provider names (e.g., ['yfinance', 'alphavantage', 'polygon'])
                    If None, uses all enabled providers
            
        Returns:
            Dictionary with validation results formatted for dashboard display
        """
        try:
            # Load local data
            from utils.data_loader import DataLoader
            loader = DataLoader()
            local_data = loader.load_ticker_ohlcv(ticker)
            
            if local_data.empty:
                return {
                    'ticker': ticker,
                    'overall_quality': 'NO DATA',
                    'consensus_confidence': 0.0,
                    'sources_compared': [],
                    'error': 'No local data found'
                }
            
            # Temporarily override enabled providers if sources specified
            original_providers = self.enabled_providers
            if sources:
                self.enabled_providers = sources
            
            # Run validation
            results = self.validate_price_data(ticker, local_data)
            
            # Restore original providers
            self.enabled_providers = original_providers
            
            # Format results for dashboard
            report = {
                'ticker': ticker,
                'overall_quality': results.get('data_quality', 'UNKNOWN'),
                'consensus_confidence': results.get('confidence_score', 0.0),
                'consensus_score': results.get('consensus_score', 0.0),
                'sources_compared': results.get('providers_used', []),
                'is_valid': results.get('is_valid', False),
                'avg_difference_pct': results.get('avg_difference_pct', 0.0),
                'max_difference_pct': results.get('max_difference_pct', 0.0),
                'dates_checked': results.get('dates_checked', 0),
                'discrepancies': results.get('discrepancies', []),
                'last_validated': results.get('last_validated')
            }
            
            # Add source comparison table
            if len(results.get('providers_used', [])) > 0:
                source_comparison = {}
                for provider in results['providers_used']:
                    source_comparison[provider] = {
                        'Status': '✅ Available',
                        'Quality': results['data_quality'],
                        'Confidence': f"{results['confidence_score']:.1f}%"
                    }
                report['source_comparison'] = source_comparison
            
            # Add validation checks table
            validation_checks = []
            validation_checks.append({
                'Check': 'Data Quality',
                'Result': results['data_quality'],
                'Status': '✅' if results['is_valid'] else '❌'
            })
            validation_checks.append({
                'Check': 'Providers Consensus',
                'Result': f"{len(results['providers_used'])} sources agree",
                'Status': '✅' if len(results['providers_used']) >= self.min_providers else '⚠️'
            })
            validation_checks.append({
                'Check': 'Average Difference',
                'Result': f"{results.get('avg_difference_pct', 0):.4f}%",
                'Status': '✅' if results.get('avg_difference_pct', 0) < 0.5 else '⚠️'
            })
            report['validation_checks'] = validation_checks
            
            return report
            
        except Exception as e:
            logger.error(f"Error in validate_ticker_multisource for {ticker}: {e}")
            return {
                'ticker': ticker,
                'overall_quality': 'ERROR',
                'consensus_confidence': 0.0,
                'sources_compared': [],
                'error': str(e)
            }


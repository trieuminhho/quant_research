"""
Streamlit Dashboard for QuantSearch Trading Platform
Interactive UI for strategy backtesting, visualization, and analysis.
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.data_loader import DataLoader, UniverseFilter
from features.base_feature import FeatureEngine, FeatureRegistry
from features.technical_indicators import *
from models.trainer import ModelTrainer
from strategies.base_strategy import MLStrategy
from backtest.portfolio import PortfolioConstructor
from backtest.engine import BacktestEngine
from data.validation import DataValidator
from ui import styles

# Configure logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="QuantSearch - Trading Strategy Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

styles.apply_dark_theme()


@st.cache_resource
def load_config():
    """Load configuration (cached)."""
    return get_config()


# Note: DataLoader is NOT cached - always create fresh instance
# to avoid stale data issues


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown("<h1 class='title-glow'>📈 QuantSearch Trading Platform</h1>", unsafe_allow_html=True)
    st.markdown("**Production-grade ML-powered strategy research and backtesting**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Data Management", "Stock Data Viewer", "Feature Engineering", "Model Training", 
         "Strategy Backtesting", "Performance Analysis", "System Configuration"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Data Management":
        show_data_management_page()
    elif page == "Stock Data Viewer":
        show_stock_viewer_page()
    elif page == "Feature Engineering":
        show_feature_engineering_page()
    elif page == "Model Training":
        show_model_training_page()
    elif page == "Strategy Backtesting":
        show_backtesting_page()
    elif page == "Performance Analysis":
        show_performance_page()
    elif page == "System Configuration":
        show_configuration_page()


def show_home_page():
    """Show home page with system overview."""
    
    st.markdown('<div class="sub-header title-glow">Welcome to QuantSearch</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**📊 Data Management**\nDownload and manage S&P 500 data")
    
    with col2:
        st.info("**🔧 Feature Engineering**\nCreate ML features from price data")
    
    with col3:
        st.info("**🤖 Model Training**\nTrain ML models with walk-forward validation")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.info("**📈 Strategy Testing**\nBacktest strategies with realistic execution")
    
    with col5:
        st.info("**📉 Performance Analysis**\nAnalyze returns, risk, and metrics")
    
    with col6:
        st.info("**⚙️ Configuration**\nManage system settings and parameters")
    
    st.markdown("---")
    
    # System status
    st.markdown('<div class="sub-header title-glow">System Status</div>', unsafe_allow_html=True)
    
    try:
        config = load_config()
        loader = DataLoader()  # Create fresh instance
        tickers = loader.get_available_tickers()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Available Stocks", len(tickers))
        
        with col2:
            features = FeatureRegistry.list_features()
            st.metric("Feature Types", len(features))
        
        with col3:
            st.metric("Data Format", "CSV")
        
        with col4:
            st.metric("Status", "✅ Operational")
        
    except Exception as e:
        st.error(f"System initialization error: {e}")


def show_data_management_page():
    """Simplified data management page - download and manage stock data with automatic validation."""
    
    st.markdown('<div class="sub-header title-glow">📊 Data Management</div>', unsafe_allow_html=True)
    st.write("Download and manage S&P 500 stock data with automatic validation.")
    
    # Initialize session state
    if 'dm_selected_tickers' not in st.session_state:
        st.session_state.dm_selected_tickers = []
    
    # Get available data
    try:
        from data.ingestion import DataIngestion
        from data.validation import DataValidator
        loader = DataLoader()
        validator = DataValidator()
        
        # Get all S&P 500 tickers
        ingestion = DataIngestion()
        all_sp500_tickers = ingestion.get_sp500_tickers()
        
        # Get downloaded tickers with metadata
        downloaded_tickers = loader.get_available_tickers()
        
        # Build status table with validation
        ticker_status = []
        for ticker in all_sp500_tickers:
            status = {
                'ticker': ticker,
                'downloaded': ticker in downloaded_tickers,
                'file_size': None,
                'records': None,
                'date_range': None,
                'last_updated': None,
                'validation_status': 'Not Downloaded'
            }
            
            if ticker in downloaded_tickers:
                try:
                    from pathlib import Path
                    filepath = Path("data/raw/ohlcv") / f"{ticker}.csv"
                    if filepath.exists():
                        # File size
                        size_bytes = filepath.stat().st_size
                        if size_bytes > 1024*1024:
                            status['file_size'] = f"{size_bytes/(1024*1024):.2f} MB"
                        else:
                            status['file_size'] = f"{size_bytes/1024:.2f} KB"
                        
                        # Last modified
                        mtime = filepath.stat().st_mtime
                        status['last_updated'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        
                        # Load data for records and date range (but don't validate on page load)
                        data = loader.load_ticker_ohlcv(ticker)
                        if data is not None:
                            status['records'] = len(data)
                            status['date_range'] = f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
                            status['validation_status'] = '📥 Downloaded'
                        else:
                            status['validation_status'] = '❌ Load Error'
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    status['validation_status'] = '⚠️ Processing Error'
            
            ticker_status.append(status)
        
        status_df = pd.DataFrame(ticker_status)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # ====== OVERVIEW =====
    st.write("### 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    downloaded_count = status_df['downloaded'].sum()
    not_downloaded = len(status_df) - downloaded_count
    total_records = sum(status_df['records'].dropna())
    
    with col1:
        st.metric("📥 Downloaded", f"{downloaded_count:,}", f"{downloaded_count/len(status_df)*100:.1f}%")
    with col2:
        st.metric("❌ Not Downloaded", f"{not_downloaded:,}", f"{not_downloaded/len(status_df)*100:.1f}%")
    with col3:
        st.metric("📊 Total Records", f"{total_records:,}")
    with col4:
        st.metric("🎯 Universe", f"{len(status_df):,}", "S&P 500")
    
    st.write("---")
    
    # ====== FILTERS, SORT & BULK SELECTION (Combined in one row) =====
    st.write("### 🛠️ Filters & Selection")
    
    # First row: Filters and Sort
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        filter_option = st.selectbox(
            "Filter",
            ["All Stocks", "Downloaded Only", "Not Downloaded", "Search..."],
            key="filter_select"
        )
    
    with col2:
        sort_option = st.selectbox(
            "Sort By",
            ["Ticker (A-Z)", "Ticker (Z-A)", "Records (High-Low)", "Last Updated"],
            key="sort_select"
        )
    
    with col3:
        if st.button("🔄 Refresh", width="stretch"):
            st.cache_resource.clear()
            st.rerun()
    
    with col4:
        show_presets_btn = st.button("⭐ Presets", width="stretch", key="show_presets")
    
    # Search bar (only if Search selected)
    search_query = ""
    if filter_option == "Search...":
        search_query = st.text_input("🔍 Search tickers", placeholder="e.g., AAPL, MSFT", key="search_input")
    
    # Quick presets (only if button clicked)
    if show_presets_btn:
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        with preset_col1:
            if st.button("Mega Caps (Top 8)", width="stretch", key="preset_mega"):
                mega_caps = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B']
                st.session_state.dm_selected_tickers = [t for t in mega_caps if t in all_sp500_tickers]
                st.rerun()
        with preset_col2:
            if st.button("Top 50", width="stretch", key="preset_top50"):
                st.session_state.dm_selected_tickers = all_sp500_tickers[:50]
                st.rerun()
        with preset_col3:
            if st.button("Tech Giants", width="stretch", key="preset_tech"):
                tech = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM']
                st.session_state.dm_selected_tickers = [t for t in tech if t in all_sp500_tickers]
                st.rerun()
        with preset_col4:
            if st.button("Clear All", width="stretch", key="preset_clear"):
                st.session_state.dm_selected_tickers = []
                st.rerun()
    
    # Second row: Bulk Selection buttons
    st.write("")  # Small spacing
    bulk_col1, bulk_col2, bulk_col3, bulk_col4 = st.columns(4)
    
    bulk_all_clicked = False
    bulk_missing_clicked = False
    bulk_invert_clicked = False
    
    with bulk_col1:
        if st.button("✅ Select All", width="stretch", key="bulk_all"):
            bulk_all_clicked = True
    
    with bulk_col2:
        if st.button("❌ Deselect All", width="stretch", key="bulk_none"):
            st.session_state.dm_selected_tickers = []
            st.rerun()
    
    with bulk_col3:
        if st.button("📥 Select Missing", width="stretch", key="bulk_missing"):
            bulk_missing_clicked = True
    
    with bulk_col4:
        if st.button("🔀 Invert Selection", width="stretch", key="bulk_invert"):
            bulk_invert_clicked = True
    
    # ====== FILTER DATA =====
    filtered_df = status_df.copy()
    
    if filter_option == "Downloaded Only":
        filtered_df = filtered_df[filtered_df['downloaded']]
    elif filter_option == "Not Downloaded":
        filtered_df = filtered_df[~filtered_df['downloaded']]
    elif filter_option == "Search..." and search_query:
        search_terms = [term.strip().upper() for term in search_query.split(',')]
        filtered_df = filtered_df[filtered_df['ticker'].str.upper().str.contains('|'.join(search_terms), na=False)]
    
    # Apply sort
    if sort_option == "Ticker (A-Z)":
        filtered_df = filtered_df.sort_values('ticker', ascending=True)
    elif sort_option == "Ticker (Z-A)":
        filtered_df = filtered_df.sort_values('ticker', ascending=False)
    elif sort_option == "Records (High-Low)":
        filtered_df = filtered_df.sort_values('records', ascending=False, na_position='last')
    elif sort_option == "Last Updated":
        filtered_df = filtered_df.sort_values('last_updated', ascending=False, na_position='last')
    
    # Handle bulk selection actions after filtering
    if bulk_all_clicked:
        st.session_state.dm_selected_tickers = filtered_df['ticker'].tolist()
        st.rerun()
    
    if bulk_missing_clicked:
        st.session_state.dm_selected_tickers = filtered_df[~filtered_df['downloaded']]['ticker'].tolist()
        st.rerun()
    
    if bulk_invert_clicked:
        current_selected = set(st.session_state.dm_selected_tickers)
        all_filtered = set(filtered_df['ticker'].tolist())
        st.session_state.dm_selected_tickers = list(all_filtered - current_selected)
        st.rerun()
    
    if st.session_state.dm_selected_tickers:
        st.info(f"✅ **{len(st.session_state.dm_selected_tickers)} ticker(s) selected**")
    
    st.write("---")
    
    # ====== DATA TABLE =====
    st.write("### 📋 Stock Inventory")
    st.write(f"Showing **{len(filtered_df)}** of **{len(status_df)}** stocks")
    
    # Create display dataframe
    display_df = filtered_df.copy()
    display_df['select'] = display_df['ticker'].isin(st.session_state.dm_selected_tickers)
    
    # Reorder columns (removed validation_status since it's now manual)
    table_columns = ['select', 'ticker', 'records', 'file_size', 'date_range', 'last_updated']
    display_df = display_df[table_columns]
    display_df.columns = ['Select', 'Ticker', 'Records', 'Size', 'Date Range', 'Last Updated']
    
    # Interactive table
    edited_df = st.data_editor(
        display_df,
        disabled=['Ticker', 'Records', 'Size', 'Date Range', 'Last Updated'],
        hide_index=True,
        width="stretch",
        height=400,
        column_config={
            'Select': st.column_config.CheckboxColumn('Select', default=False, width='small'),
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Records': st.column_config.NumberColumn('Records', format='%d', width='small'),
            'Size': st.column_config.TextColumn('Size', width='small'),
            'Date Range': st.column_config.TextColumn('Date Range', width='medium'),
            'Last Updated': st.column_config.TextColumn('Last Updated', width='medium')
        }
    )
    
    # Update selection from table
    st.session_state.dm_selected_tickers = edited_df[edited_df['Select']]['Ticker'].tolist()
    
    st.write("---")
    
    # ====== ACTIONS: DOWNLOAD, DELETE, VALIDATE (Combined) =====
    if st.session_state.dm_selected_tickers:
        st.write("### ⚡ Actions")
        
        selected_count = len(st.session_state.dm_selected_tickers)
        selected_downloaded = [t for t in st.session_state.dm_selected_tickers if t in downloaded_tickers]
        selected_not_downloaded = [t for t in st.session_state.dm_selected_tickers if t not in downloaded_tickers]
        
        # Show selection metrics
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Selected", selected_count)
        with info_col2:
            st.metric("Already Downloaded", len(selected_downloaded))
        with info_col3:
            st.metric("To Download", len(selected_not_downloaded))
        
        st.write("")  # Spacing
        
        # Action buttons in tabs for cleaner organization
        tab1, tab2, tab3 = st.tabs(["📥 Download", "🗑️ Delete", "✅ Validate"])
        
        # TAB 1: DOWNLOAD
        with tab1:
            if selected_not_downloaded:
                st.write(f"**Download {len(selected_not_downloaded)} stock(s)**")
                
                dl_col1, dl_col2 = st.columns([2, 1])
                with dl_col1:
                    date_range = st.selectbox(
                        "Time Range",
                        ["1 Year", "3 Years", "5 Years", "10 Years", "Max (2010+)"],
                        index=3,
                        key="download_range"
                    )
                with dl_col2:
                    max_workers = st.number_input("Threads", 1, 10, 5, key="download_workers")
                
                # Calculate date range
                end_date = datetime.now()
                if date_range == "1 Year":
                    start_date = end_date - timedelta(days=365)
                elif date_range == "3 Years":
                    start_date = end_date - timedelta(days=365*3)
                elif date_range == "5 Years":
                    start_date = end_date - timedelta(days=365*5)
                elif date_range == "10 Years":
                    start_date = end_date - timedelta(days=365*10)
                else:  # Max
                    start_date = datetime(2010, 1, 1)
                
                if st.button(f"📥 Download Now", type="primary", width="stretch", key="download_btn"):
                    with st.spinner(f"Downloading {len(selected_not_downloaded)} stocks..."):
                        try:
                            ingestion = DataIngestion(
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d')
                            )
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Download stocks
                            status_text.text("⏳ Downloading data...")
                            results = ingestion.download_all_tickers(
                                selected_not_downloaded, 
                                max_workers=max_workers
                            )
                            
                            progress_bar.progress(50)
                            
                            successful = sum(1 for v in results.values() if v)
                            failed = len(results) - successful
                            successful_tickers = [k for k, v in results.items() if v]
                            
                            # Automatic validation of downloaded stocks
                            if successful_tickers:
                                status_text.text("🔍 Validating downloaded data...")
                                
                                validation_results = {}
                                for i, ticker in enumerate(successful_tickers):
                                    try:
                                        data = loader.load_ticker_ohlcv(ticker)
                                        if data is not None:
                                            validation_results[ticker] = validator.validate_price_data(ticker, data)
                                        progress_bar.progress(50 + int((i + 1) / len(successful_tickers) * 50))
                                    except Exception as e:
                                        validation_results[ticker] = {
                                            'is_valid': False,
                                            'errors': {'load_error': [str(e)]}
                                        }
                                
                                progress_bar.progress(100)
                                status_text.empty()
                                
                                # Show results
                                valid_count = sum(1 for v in validation_results.values() if v['is_valid'])
                                issues_count = len(validation_results) - valid_count
                                
                                if failed == 0 and issues_count == 0:
                                    st.success(f"✅ Downloaded and validated all {successful} stocks successfully! All data is valid.")
                                elif failed == 0 and issues_count > 0:
                                    st.warning(f"⚠️ Downloaded {successful} stocks: {valid_count} valid, {issues_count} with issues")
                                else:
                                    st.warning(f"⚠️ Downloaded {successful} stocks ({valid_count} valid, {issues_count} with issues), {failed} failed")
                                
                                # Show detailed validation results
                                if issues_count > 0:
                                    with st.expander(f"📋 View Detailed Validation Results ({issues_count} stocks with issues)", expanded=True):
                                        for ticker, result in validation_results.items():
                                            if not result['is_valid']:
                                                st.markdown(f"### 🔍 {ticker}")
                                                
                                                # Error summary
                                                error_count = sum(len(errors) for errors in result.get('errors', {}).values() if errors)
                                                st.markdown(f'<div class="validation-error">**{error_count} validation error(s) found**</div>', unsafe_allow_html=True)
                                                
                                                # Show errors by category
                                                for category, errors in result.get('errors', {}).items():
                                                    if errors:
                                                        st.markdown(f"**❌ {category.replace('_', ' ').title()}:**")
                                                        for error in errors:
                                                            st.markdown(f"  - {error}")
                                                
                                                # Show warnings if any
                                                warnings = result.get('warnings', {})
                                                if warnings:
                                                    warning_count = sum(len(w) for w in warnings.values() if w)
                                                    if warning_count > 0:
                                                        st.markdown(f"**⚠️ {warning_count} warning(s):**")
                                                        for category, warn_list in warnings.items():
                                                            if warn_list:
                                                                st.markdown(f"  - {category.replace('_', ' ').title()}: {', '.join(map(str, warn_list))}")
                                                
                                                st.markdown("---")
                                
                                # Show all validation results in expandable section
                                with st.expander(f"📊 View All Validation Details ({len(validation_results)} stocks)", expanded=False):
                                    for ticker, result in validation_results.items():
                                        if result['is_valid']:
                                            st.markdown(f"### ✅ {ticker}")
                                            st.markdown(f'<div class="validation-success">**All validation checks passed**</div>', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"### ❌ {ticker}")
                                            st.markdown(f'<div class="validation-error">**Validation failed**</div>', unsafe_allow_html=True)
                                        
                                        # Show metadata
                                        metadata = result.get('metadata', {})
                                        if metadata:
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Records", metadata.get('record_count', 'N/A'))
                                            with col2:
                                                st.metric("Date Range", f"{metadata.get('date_range_days', 'N/A')} days")
                                            with col3:
                                                completeness = metadata.get('completeness', 0)
                                                st.metric("Completeness", f"{completeness:.1%}")
                                            with col4:
                                                st.metric("File Size", metadata.get('file_size_mb', 'N/A'))
                                        
                                        st.markdown("---")
                            else:
                                progress_bar.progress(100)
                                status_text.empty()
                                st.warning(f"⚠️ Downloaded {successful} stocks, {failed} failed")
                            
                            # Auto-refresh
                            time.sleep(2)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Download error: {e}")
            else:
                st.info("💡 All selected stocks are already downloaded.")
        
        # TAB 2: DELETE
        with tab2:
            if selected_downloaded:
                st.write(f"**Delete {len(selected_downloaded)} downloaded stock(s)**")
                st.warning("⚠️ This will permanently delete the CSV files!")
                
                if st.button(f"🗑️ Delete {len(selected_downloaded)} File(s)", 
                           type="secondary", width="stretch", key="delete_btn"):
                    try:
                        from pathlib import Path
                        deleted = 0
                        for ticker in selected_downloaded:
                            filepath = Path("data/raw/ohlcv") / f"{ticker}.csv"
                            if filepath.exists():
                                filepath.unlink()
                                deleted += 1
                        
                        st.success(f"✅ Deleted {deleted} file(s)")
                        st.session_state.dm_selected_tickers = []
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Delete error: {e}")
            else:
                st.info("💡 No downloaded stocks selected to delete.")
        
        # TAB 3: VALIDATE
        with tab3:
            if selected_downloaded:
                st.write(f"**Validate {len(selected_downloaded)} downloaded stock(s)**")
                st.info("💡 Run data quality validation on selected stocks")
                
                if st.button(f"✅ Validate {len(selected_downloaded)} Stock(s)", 
                           type="primary", width="stretch", key="validate_btn"):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        validation_results = {}
                        for i, ticker in enumerate(selected_downloaded):
                            try:
                                status_text.text(f"Validating {ticker}... ({i+1}/{len(selected_downloaded)})")
                                data = loader.load_ticker_ohlcv(ticker)
                                if data is not None:
                                    validation_results[ticker] = validator.validate_price_data(ticker, data)
                                progress_bar.progress(int((i + 1) / len(selected_downloaded) * 100))
                            except Exception as e:
                                validation_results[ticker] = {
                                    'is_valid': False,
                                    'errors': {'load_error': [str(e)]}
                                }
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        
                        # Show summary
                        valid_count = sum(1 for v in validation_results.values() if v['is_valid'])
                        issues_count = len(validation_results) - valid_count
                        
                        if issues_count == 0:
                            st.success(f"✅ All {len(validation_results)} stocks validated successfully!")
                        else:
                            st.warning(f"⚠️ Validation complete: {valid_count} valid, {issues_count} with issues")
                        
                        # Show detailed results
                        with st.expander(f"� View Validation Results", expanded=True):
                            for ticker, result in validation_results.items():
                                if result['is_valid']:
                                    st.markdown(f"### ✅ {ticker} - Valid")
                                else:
                                    st.markdown(f"### ❌ {ticker} - Issues Found")
                                    for category, errors in result.get('errors', {}).items():
                                        if errors:
                                            st.markdown(f"**{category.replace('_', ' ').title()}:**")
                                            for error in errors:
                                                st.markdown(f"  - {error}")
                                st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"❌ Validation error: {e}")
            else:
                st.info("� No downloaded stocks selected to validate.")
    
    else:
        st.info("💡 **Tip:** Select stocks using checkboxes or bulk actions to download/manage data.")




def show_stock_viewer_page():
    """Interactive stock data viewer with advanced charting and company analysis."""
    
    st.markdown('<div class="sub-header title-glow">📊 Stock Data Viewer</div>', unsafe_allow_html=True)
    st.write("View and analyze downloaded stock data with professional charts and comprehensive company information.")
    
    # Get available data
    loader = DataLoader()
    downloaded_tickers = loader.get_available_tickers()
    
    if not downloaded_tickers:
        st.warning("⚠️ No stock data available. Please download data from the Data Management page first.")
        return
    
    # Stock selector
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_ticker = st.selectbox(
            "Select Stock",
            sorted(downloaded_tickers),
            key="viewer_ticker"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🔄 Refresh Data", width="stretch"):
            st.rerun()
    
    if not selected_ticker:
        return
    
    # Load stock data
    data = loader.load_ticker_ohlcv(selected_ticker)
    
    if data is None:
        st.error(f"❌ Could not load data for {selected_ticker}")
        return
    
    # Fetch company info from yfinance
    try:
        import yfinance as yf
        ticker_obj = yf.Ticker(selected_ticker)
        info = ticker_obj.info
    except Exception as e:
        st.warning(f"Could not fetch company info: {e}")
        info = {}
    
    # ===== COMPANY HEADER =====
    st.write("---")
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        company_name = info.get('longName', selected_ticker)
        st.markdown(f"### {company_name}")
        st.markdown(f"**{selected_ticker}** • {info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}")
    
    # ===== KEY METRICS =====
    price_col = 'adj_close' if 'adj_close' in data.columns else 'close'
    current_price = data[price_col].iloc[-1]
    prev_price = data[price_col].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0
    
    met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
    with met_col1:
        st.metric("Current Price", f"${current_price:.2f}", 
                 f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
    with met_col2:
        mkt_cap = info.get('marketCap', 0)
        if mkt_cap > 1e12:
            st.metric("Market Cap", f"${mkt_cap/1e12:.2f}T")
        elif mkt_cap > 1e9:
            st.metric("Market Cap", f"${mkt_cap/1e9:.2f}B")
        else:
            st.metric("Market Cap", "N/A")
    with met_col3:
        st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A")
    with met_col4:
        st.metric("EPS", f"${info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else "N/A")
    with met_col5:
        div_yield = info.get('dividendYield', 0)
        # Fix: dividendYield is already a decimal (0.0042 = 0.42%), so multiply by 100
        if div_yield and div_yield > 0:
            st.metric("Div Yield", f"{div_yield * 100:.2f}%")
        else:
            st.metric("Div Yield", "N/A")
    
    st.write("---")
    
    # ===== CHART CONTROLS =====
    st.write("### 📈 Price Chart")
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Area"], key="viewer_chart_type")
    with chart_col2:
        show_ma = st.multiselect("Moving Averages", ["MA20", "MA50"], default=["MA20", "MA50"], key="viewer_ma")
    with chart_col3:
        period = st.selectbox("Period", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Max"], index=4, key="viewer_period")
    
    # Filter data by period
    end_date_filter = data.index[-1]
    if period == "1M":
        start_date_filter = end_date_filter - timedelta(days=30)
    elif period == "3M":
        start_date_filter = end_date_filter - timedelta(days=90)
    elif period == "6M":
        start_date_filter = end_date_filter - timedelta(days=180)
    elif period == "YTD":
        start_date_filter = pd.Timestamp(end_date_filter.year, 1, 1)
    elif period == "1Y":
        start_date_filter = end_date_filter - timedelta(days=365)
    elif period == "3Y":
        start_date_filter = end_date_filter - timedelta(days=365*3)
    elif period == "5Y":
        start_date_filter = end_date_filter - timedelta(days=365*5)
    else:  # Max
        start_date_filter = data.index[0]
    
    chart_data = data[data.index >= start_date_filter].copy()
    
    # ===== CREATE PRICE CHART =====
    fig = go.Figure()
    
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=chart_data.index,
            open=chart_data['open'],
            high=chart_data['high'],
            low=chart_data['low'],
            close=chart_data['close'],
            name=selected_ticker
        ))
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data[price_col],
            mode='lines',
            name=selected_ticker,
            line=dict(color='#8B5CF6', width=2)
        ))
    else:  # Area
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data[price_col],
            mode='lines',
            name=selected_ticker,
            fill='tozeroy',
            line=dict(color='#8B5CF6', width=2)
        ))
    
    # Add moving averages
    if "MA20" in show_ma:
        chart_data['ma20'] = chart_data[price_col].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['ma20'],
            mode='lines',
            name='MA20',
            line=dict(color='#22D3EE', width=1)
        ))
    
    if "MA50" in show_ma:
        chart_data['ma50'] = chart_data[price_col].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['ma50'],
            mode='lines',
            name='MA50',
            line=dict(color='#F59E0B', width=1)
        ))
    
    fig.update_layout(
        title=f"{selected_ticker} - {period} Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # ===== VOLUME CHART =====
    st.write("### 📊 Volume")
    fig_vol = go.Figure()
    colors = ['red' if row['close'] < row['open'] else 'green' 
             for idx, row in chart_data.iterrows()]
    
    fig_vol.add_trace(go.Bar(
        x=chart_data.index,
        y=chart_data['volume'],
        name='Volume',
        marker_color=colors
    ))
    
    fig_vol.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        height=200,
        margin=dict(t=20, b=40),
        template='plotly_dark',
        showlegend=False
    )
    
    st.plotly_chart(fig_vol, width="stretch")
    
    # ===== COMPANY INFO TABS =====
    st.write("---")
    st.write("### 📋 Company Information")
    
    info_tabs = st.tabs(["📊 Financials", "💰 Valuation", "📈 Performance", "ℹ️ Company Info"])
    
    with info_tabs[0]:
        fin_col1, fin_col2 = st.columns(2)
        with fin_col1:
            st.write("**Profitability**")
            st.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "N/A")
            st.metric("Operating Margin", f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else "N/A")
            st.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A")
            st.metric("ROA", f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else "N/A")
        
        with fin_col2:
            st.write("**Revenue & Earnings**")
            revenue = info.get('totalRevenue', 0)
            st.metric("Revenue", f"${revenue/1e9:.2f}B" if revenue else "N/A")
            st.metric("Revenue Growth", f"{info.get('revenueGrowth', 0)*100:.2f}%" if info.get('revenueGrowth') else "N/A")
            st.metric("Earnings Growth", f"{info.get('earningsGrowth', 0)*100:.2f}%" if info.get('earningsGrowth') else "N/A")
            st.metric("EPS", f"${info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else "N/A")
    
    with info_tabs[1]:
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.write("**Price Ratios**")
            st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A")
            st.metric("Forward P/E", f"{info.get('forwardPE', 0):.2f}" if info.get('forwardPE') else "N/A")
            st.metric("PEG Ratio", f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else "N/A")
            st.metric("Price/Book", f"{info.get('priceToBook', 0):.2f}" if info.get('priceToBook') else "N/A")
        
        with val_col2:
            st.write("**Market Data**")
            mkt_cap = info.get('marketCap', 0)
            if mkt_cap > 1e12:
                st.metric("Market Cap", f"${mkt_cap/1e12:.2f}T")
            elif mkt_cap > 1e9:
                st.metric("Market Cap", f"${mkt_cap/1e9:.2f}B")
            else:
                st.metric("Market Cap", "N/A")
            
            ent_value = info.get('enterpriseValue', 0)
            if ent_value > 1e12:
                st.metric("Enterprise Value", f"${ent_value/1e12:.2f}T")
            elif ent_value > 1e9:
                st.metric("Enterprise Value", f"${ent_value/1e9:.2f}B")
            else:
                st.metric("Enterprise Value", "N/A")
            
            st.metric("Price/Sales", f"{info.get('priceToSalesTrailing12Months', 0):.2f}" if info.get('priceToSalesTrailing12Months') else "N/A")
            st.metric("EV/Revenue", f"{info.get('enterpriseToRevenue', 0):.2f}" if info.get('enterpriseToRevenue') else "N/A")
    
    with info_tabs[2]:
        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            st.write("**Growth Metrics**")
            st.metric("Revenue Growth", f"{info.get('revenueGrowth', 0)*100:.2f}%" if info.get('revenueGrowth') else "N/A")
            st.metric("Earnings Growth", f"{info.get('earningsGrowth', 0)*100:.2f}%" if info.get('earningsGrowth') else "N/A")
            st.metric("Earnings Quarterly Growth", f"{info.get('earningsQuarterlyGrowth', 0)*100:.2f}%" if info.get('earningsQuarterlyGrowth') else "N/A")
        
        with perf_col2:
            st.write("**Cash Flow & Balance Sheet**")
            fcf = info.get('freeCashflow', 0)
            st.metric("Free Cash Flow", f"${fcf/1e9:.2f}B" if fcf else "N/A")
            st.metric("Operating Cash Flow", f"${info.get('operatingCashflow', 0)/1e9:.2f}B" if info.get('operatingCashflow') else "N/A")
            st.metric("Current Ratio", f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else "N/A")
            st.metric("Debt/Equity", f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else "N/A")
    
    with info_tabs[3]:
        st.write("**Company Overview**")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Website:** {info.get('website', 'N/A')}")
        st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "**Employees:** N/A")
        
        if info.get('longBusinessSummary'):
            st.write("**Description:**")
            st.write(info.get('longBusinessSummary'))
    
    # ===== DATA SUMMARY =====
    st.write("---")
    st.write("### 📊 Data Summary")
    sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
    
    total_return = ((current_price - data[price_col].iloc[0]) / data[price_col].iloc[0] * 100)
    
    with sum_col1:
        st.metric("Total Records", f"{len(data):,}")
    with sum_col2:
        st.metric("Date Range", f"{(data.index[-1] - data.index[0]).days} days")
    with sum_col3:
        st.metric("First Price", f"${data[price_col].iloc[0]:.2f}")
    with sum_col4:
        st.metric("Total Return", f"{total_return:+.2f}%")
    with sum_col5:
        st.metric("Avg Volume", f"{data['volume'].mean()/1e6:.2f}M")



def show_data_validation_page():
    """Data validation page to cross-check data quality."""
    
    st.markdown('<div class="sub-header title-glow">Data Validation & Quality Control</div>', unsafe_allow_html=True)
    st.write("Cross-validate stock data against yfinance to ensure accuracy before analysis.")
    
    # Create DataLoader and Validator
    loader = DataLoader()
    validator = DataValidator()
    
    # Get available tickers
    available_tickers = loader.get_available_tickers()
    
    if not available_tickers:
        st.warning("No data files found. Please download data first in the Data Management page.")
        return
    
    st.write(f"**Available Tickers:** {len(available_tickers)}")
    
    # Validation options
    st.write("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        validation_mode = st.radio(
            "Validation Mode",
            ["Quick Check (5 tickers)", "All Tickers", "Selected Tickers"],
            help="Quick Check validates 5 random tickers. All Tickers validates your entire dataset."
        )
    
    with col2:
        tolerance = st.slider(
            "Price Tolerance (%)",
            min_value=0.1,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Maximum acceptable price difference for validation"
        )
    
    # Ticker selection for "Selected Tickers" mode
    selected_tickers = []
    if validation_mode == "Selected Tickers":
        selected_tickers = st.multiselect(
            "Select tickers to validate",
            available_tickers,
            default=available_tickers[:5]
        )
    
    # Run validation button
    if st.button("🔍 Run Validation", type="primary"):
        
        # Determine which tickers to validate
        if validation_mode == "Quick Check (5 tickers)":
            import random
            tickers_to_validate = random.sample(available_tickers, min(5, len(available_tickers)))
        elif validation_mode == "All Tickers":
            tickers_to_validate = available_tickers
        else:
            tickers_to_validate = selected_tickers
        
        if not tickers_to_validate:
            st.warning("Please select at least one ticker to validate.")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for idx, ticker in enumerate(tickers_to_validate):
            status_text.text(f"Validating {ticker}... ({idx+1}/{len(tickers_to_validate)})")
            
            try:
                local_data = loader.load_ticker_ohlcv(ticker)
                if local_data is not None:
                    result = validator.validate_price_data(ticker, local_data, tolerance=tolerance/100)
                    results.append(result)
            except Exception as e:
                st.warning(f"Error validating {ticker}: {e}")
            
            progress_bar.progress((idx + 1) / len(tickers_to_validate))
        
        status_text.empty()
        progress_bar.empty()
        
        if not results:
            st.error("No validation results available.")
            return
        
        # Convert to DataFrame
        validation_df = pd.DataFrame(results)
        
        # Store in session state
        st.session_state.validation_results = validation_df
        
        # Display summary
        st.write("---")
        st.write("### Validation Summary")
        
        summary = validator.get_validation_summary(validation_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tickers", summary['total_tickers'])
        
        with col2:
            st.metric("Valid Tickers", summary['valid_tickers'])
        
        with col3:
            st.metric("Validation Rate", f"{summary['validation_rate']:.1f}%")
        
        with col4:
            st.metric("Avg Confidence", f"{summary['avg_confidence']:.1f}%")
        
        # Quality distribution
        st.write("### Data Quality Distribution")
        quality_dist = validation_df['data_quality'].value_counts().sort_index()
        
        # Color map for quality
        quality_colors = {
            'EXCELLENT': '#10B981',
            'GOOD': '#22D3EE',
            'ACCEPTABLE': '#F59E0B',
            'POOR': '#EF4444',
            'STALE': '#94A3B8',
            'MISSING': '#1F2937',
            'ERROR': '#EF4444',
            'UNVERIFIED': '#8B5CF6'
        }
        
        fig_quality = go.Figure(data=[
            go.Bar(
                x=quality_dist.index,
                y=quality_dist.values,
                marker_color=[quality_colors.get(q, '#8B5CF6') for q in quality_dist.index],
                text=quality_dist.values,
                textposition='auto'
            )
        ])
        
        fig_quality.update_layout(
            title="Data Quality Distribution",
            xaxis_title="Quality Level",
            yaxis_title="Number of Tickers",
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_quality, width="stretch")
        
        # Detailed results table
        st.write("### Detailed Validation Results")
        
        # Add filter
        quality_filter = st.multiselect(
            "Filter by Quality",
            options=validation_df['data_quality'].unique().tolist(),
            default=validation_df['data_quality'].unique().tolist()
        )
        
        filtered_df = validation_df[validation_df['data_quality'].isin(quality_filter)].copy()
        
        # Format display
        display_df = filtered_df[[
            'ticker', 'data_quality', 'confidence_score', 'is_valid',
            'dates_checked', 'avg_difference_pct', 'max_difference_pct'
        ]].copy() if 'dates_checked' in filtered_df.columns else filtered_df[[
            'ticker', 'data_quality', 'confidence_score', 'is_valid'
        ]].copy()
        
        # Rename columns for display
        display_df.columns = display_df.columns.str.replace('_', ' ').str.title()
        
        # Color code the rows
        def highlight_quality(row):
            quality = row['Data Quality']
            if quality in ['EXCELLENT', 'GOOD']:
                return ['background-color: rgba(16,185,129,0.18)'] * len(row)
            elif quality == 'ACCEPTABLE':
                return ['background-color: rgba(245,158,11,0.18)'] * len(row)
            elif quality in ['POOR', 'ERROR']:
                return ['background-color: rgba(239,68,68,0.18)'] * len(row)
            else:
                return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_quality, axis=1),
            width="stretch",
            height=400
        )
        
        # Show discrepancies for problematic tickers
        problem_tickers = validation_df[
            (validation_df['data_quality'].isin(['POOR', 'ACCEPTABLE'])) & 
            (validation_df['discrepancies'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
        ]
        
        if not problem_tickers.empty:
            st.write("---")
            st.write("### ⚠️ Tickers with Discrepancies")
            
            for _, row in problem_tickers.iterrows():
                with st.expander(f"{row['ticker']} - {row['data_quality']} (Confidence: {row['confidence_score']:.1f}%)"):
                    if isinstance(row['discrepancies'], list) and row['discrepancies']:
                        discrepancies_df = pd.DataFrame(row['discrepancies'])
                        st.dataframe(discrepancies_df, width="stretch")
                        
                        # Recommendation
                        st.warning(
                            f"**Recommendation:** Consider re-downloading {row['ticker']} data. "
                            f"Current data shows {row.get('avg_difference_pct', 0):.2f}% average price difference from yfinance."
                        )
        
        # Action buttons
        st.write("---")
        st.write("### Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Validation Report"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_path = f"validation_report_{timestamp}.csv"
                validation_df.to_csv(report_path, index=False)
                st.success(f"Report exported to: {report_path}")
        
        with col2:
            problem_list = validation_df[validation_df['data_quality'].isin(['POOR', 'ERROR', 'STALE'])]['ticker'].tolist()
            if problem_list and st.button("🔄 Re-download Problem Tickers"):
                st.info(f"Would re-download: {', '.join(problem_list)}")
                st.info("Navigate to Data Management page to re-download these tickers.")
    
    # Display previous results if available
    elif hasattr(st.session_state, 'validation_results') and st.session_state.validation_results is not None:
        st.info("📊 Showing previous validation results. Click 'Run Validation' to refresh.")
        
        validation_df = st.session_state.validation_results
        summary = validator.get_validation_summary(validation_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tickers", summary['total_tickers'])
        
        with col2:
            st.metric("Valid Tickers", summary['valid_tickers'])
        
        with col3:
            st.metric("Validation Rate", f"{summary['validation_rate']:.1f}%")
        
        with col4:
            st.metric("Avg Confidence", f"{summary['avg_confidence']:.1f}%")
    
    # ===== MULTI-SOURCE VALIDATION =====
    st.write("---")
    st.markdown('<div class="sub-header title-glow">🔍 Multi-Source Validation (Advanced)</div>', unsafe_allow_html=True)
    st.write("Cross-validate individual tickers against multiple data providers for maximum confidence.")
    
    ms_col1, ms_col2 = st.columns([2, 1])
    
    with ms_col1:
        ms_ticker = st.selectbox(
            "Select Ticker for Multi-Source Validation",
            available_tickers,
            key="ms_ticker"
        )
    
    with ms_col2:
        st.write("")
        st.write("")
        ms_sources = st.multiselect(
            "Data Sources",
            ["yfinance", "finnhub"],
            default=["yfinance"],
            help="Finnhub: 60 calls/min free tier - Get API key from finnhub.io",
            key="ms_sources"
        )
    
    if st.button("🔍 Run Multi-Source Validation", type="primary", key="ms_validate"):
        if not ms_ticker:
            st.warning("Please select a ticker.")
        elif not ms_sources:
            st.warning("Please select at least one data source.")
        else:
            with st.spinner(f"Validating {ms_ticker} against {len(ms_sources)} source(s)..."):
                try:
                    # Run multi-source validation
                    ms_result = validator.validate_ticker_multisource(ms_ticker, ms_sources)
                    
                    # Display overall quality
                    st.write("---")
                    st.markdown(f"### 📊 Validation Results for **{ms_ticker}**")
                    
                    quality = ms_result['overall_quality']
                    confidence = ms_result['consensus_confidence']
                    
                    # Quality badge
                    if quality in ['EXCELLENT', 'GOOD']:
                        st.markdown(f'<div class="validation-success">✅ **Quality: {quality}** • Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
                    elif quality == 'ACCEPTABLE':
                        st.markdown(f'<div class="validation-warning">⚠️ **Quality: {quality}** • Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="validation-error">❌ **Quality: {quality}** • Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
                    
                    # Key metrics
                    ms_met_col1, ms_met_col2, ms_met_col3, ms_met_col4 = st.columns(4)
                    
                    with ms_met_col1:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with ms_met_col2:
                        st.metric("Sources Used", len(ms_result['sources_compared']))
                    with ms_met_col3:
                        avg_diff = ms_result.get('avg_difference_pct', 0)
                        st.metric("Avg Difference", f"{avg_diff:.4f}%")
                    with ms_met_col4:
                        max_diff = ms_result.get('max_difference_pct', 0)
                        st.metric("Max Difference", f"{max_diff:.4f}%")
                    
                    # Source comparison table
                    st.write("---")
                    st.markdown("#### 📡 Source Comparison")
                    
                    if 'source_comparison' in ms_result and ms_result['source_comparison']:
                        source_df = pd.DataFrame(ms_result['source_comparison']).T
                        st.dataframe(source_df, width="stretch")
                    else:
                        sources_used = ms_result['sources_compared']
                        if sources_used:
                            st.success(f"✅ Successfully validated against: {', '.join(sources_used)}")
                        else:
                            st.error("❌ No sources were available for validation")
                    
                    # Validation checks table
                    st.write("---")
                    st.markdown("#### ✓ Validation Checks")
                    
                    if 'validation_checks' in ms_result:
                        checks_df = pd.DataFrame(ms_result['validation_checks'])
                        st.dataframe(checks_df, width="stretch")
                    
                    # Discrepancies
                    if 'discrepancies' in ms_result and ms_result['discrepancies']:
                        st.write("---")
                        st.markdown("#### ⚠️ Discrepancies Found")
                        for disc in ms_result['discrepancies']:
                            st.markdown(f"- {disc}")
                    
                    # Error handling
                    if 'error' in ms_result:
                        st.error(f"❌ Validation error: {ms_result['error']}")
                    
                    # Recommendations
                    st.write("---")
                    st.markdown("#### 💡 Recommendations")
                    
                    if quality in ['EXCELLENT', 'GOOD']:
                        st.success("✅ Data quality is excellent. Safe to use for analysis.")
                    elif quality == 'ACCEPTABLE':
                        st.warning("⚠️ Data quality is acceptable but consider monitoring. Small discrepancies detected.")
                    elif quality == 'POOR':
                        st.error("❌ Data quality is poor. Recommend re-downloading this ticker.")
                        if st.button(f"🔄 Re-download {ms_ticker}", key="ms_redownload"):
                            st.info("Navigate to Data Management page to re-download.")
                    elif quality == 'ERROR':
                        st.error("❌ Validation failed. Check data integrity or API keys.")
                    elif quality == 'NO DATA':
                        st.warning("⚠️ No local data found. Download data first.")
                    
                    # Detailed stats
                    with st.expander("📈 View Detailed Statistics"):
                        st.json(ms_result)
                    
                except Exception as e:
                    st.error(f"❌ Multi-source validation error: {e}")
                    logger.exception(f"Multi-source validation failed for {ms_ticker}")


def show_configuration_page():
    """System configuration page."""
    
    st.markdown('<div class="sub-header title-glow">System Configuration</div>', unsafe_allow_html=True)
    
    try:
        config = load_config()
        
        st.write("### Current Configuration")
        
        # Show key settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Settings:**")
            st.code(f"Start Date: {config.data_start_date}")
            st.code(f"Data Format: CSV")
            
            st.write("**Model Settings:**")
            st.code(f"Default Model: {config.model_type}")
            st.code(f"Max Positions: {config.max_positions}")
        
        with col2:
            st.write("**Backtest Settings:**")
            st.code(f"Initial Capital: ${config.initial_capital:,}")
            st.code(f"Commission: {config.commission_pct:.3%}")
            st.code(f"Slippage: {config.slippage_bps} bps")
            
            st.write("**Paths:**")
            st.code(f"Data Root: {config.get_path('raw_data')}")
        
        st.write("---")
        st.info("Edit `config.yaml` to modify system settings")
        
    except Exception as e:
        st.error(f"Error loading configuration: {e}")


if __name__ == "__main__":
    main()

"""
Data Management Page
Extracted from main dashboard for modular architecture.
"""

"""
Streamlit Dashboard for QuantSearch Trading Platform
Interactive UI for strategy backtesting, visualization, and analysis.
"""

import sys
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
from data.validation import DataValidator
from ui import styles

# Configure logging
logger = logging.getLogger(__name__)

# Validation cache file
VALIDATION_CACHE_FILE = Path("data/raw/ohlcv/validation_cache.json")

def load_validation_cache():
    """Load validation results from disk cache."""
    if VALIDATION_CACHE_FILE.exists():
        try:
            with open(VALIDATION_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading validation cache: {e}")
            return {}
    return {}

def save_validation_cache(cache):
    """Save validation results to disk cache."""
    try:
        VALIDATION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VALIDATION_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving validation cache: {e}")

# Page configuration
st.set_page_config(
    page_title="QuantSearch - Trading Strategy Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

styles.apply_dark_theme()

# Note: DataLoader is NOT cached - always create fresh instance
# to avoid stale data issues


# Rename function to show() for consistency
def show():
    """Simplified data management page - download and manage stock data with automatic validation."""
    
    st.markdown('<div class="sub-header title-glow">📊 Data Management</div>', unsafe_allow_html=True)
    st.write("Download and manage S&P 500 stock data with automatic validation.")
    
    # Initialize session state
    if 'dm_selected_tickers' not in st.session_state:
        st.session_state.dm_selected_tickers = []
    if 'dm_validation_results' not in st.session_state:
        # Load validation results from disk cache
        st.session_state.dm_validation_results = load_validation_cache()
    
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
                'validation_status': '❌'  # Default: not downloaded
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
                        
                        # Load data for records and date range
                        data = loader.load_ticker_ohlcv(ticker)
                        if data is not None:
                            status['records'] = len(data)
                            status['date_range'] = f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
                            
                            # Check if we have validation results in session state
                            if ticker in st.session_state.dm_validation_results:
                                val_result = st.session_state.dm_validation_results[ticker]
                                if val_result.get('is_valid'):
                                    quality = val_result.get('data_quality', 'UNKNOWN')
                                    if quality == 'EXCELLENT':
                                        status['validation_status'] = '✅'
                                    elif quality == 'GOOD':
                                        status['validation_status'] = '✅'
                                    elif quality == 'ACCEPTABLE':
                                        status['validation_status'] = '⚠️'
                                    else:
                                        status['validation_status'] = '❌'
                                else:
                                    status['validation_status'] = '❌'
                            else:
                                status['validation_status'] = '➖'  # Not yet validated
                        else:
                            status['validation_status'] = '❌'
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    status['validation_status'] = '❌'
            
            ticker_status.append(status)
        
        status_df = pd.DataFrame(ticker_status)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # ====== OVERVIEW (No title) =====
    col1, col2, col3 = st.columns(3)
    
    downloaded_count = status_df['downloaded'].sum()
    total_records = sum(status_df['records'].dropna())
    universe_size = len(status_df)
    
    with col1:
        st.metric("📥 Downloaded", f"{downloaded_count}/{universe_size}")
    with col2:
        st.metric("📊 Total Records", f"{total_records:,}")
    with col3:
        universe_option = st.selectbox("🎯 Universe", ["S&P 500"], key="universe_select")
    
    st.write("---")
    
    # ====== FILTERS & BULK SELECTION =====
    
    # Row 1: Filter and Sort controls
    col1, col2, col3 = st.columns([3, 3, 2])
    
    with col1:
        filter_option = st.selectbox(
            "🔍 Filter",
            ["All Stocks", "Downloaded Only", "Search..."],
            key="filter_select"
        )
    
    with col2:
        sort_option = st.selectbox(
            "📊 Sort",
            ["Ticker (A-Z)", "Ticker (Z-A)", "Records (High-Low)", "Last Updated"],
            key="sort_select"
        )
    
    with col3:
        bulk_action = st.selectbox(
            "⚡ Bulk Actions",
            ["❌ Deselect All", "✅ Select All", "📥 Select Missing", "🔀 Invert Selection"],
            key="bulk_action_select"
        )
    
    # Handle bulk action selection
    bulk_all_clicked = False
    bulk_missing_clicked = False
    bulk_invert_clicked = False
    
    if bulk_action == "✅ Select All":
        bulk_all_clicked = True
    elif bulk_action == "❌ Deselect All":
        # Only rerun if there are actually selections to clear
        if st.session_state.dm_selected_tickers:
            st.session_state.dm_selected_tickers = []
            st.rerun()
    elif bulk_action == "📥 Select Missing":
        bulk_missing_clicked = True
    elif bulk_action == "🔀 Invert Selection":
        bulk_invert_clicked = True
    
    # Search bar (only if Search selected)
    search_query = ""
    if filter_option == "Search...":
        search_query = st.text_input("🔍 Search tickers", placeholder="e.g., AAPL, MSFT", key="search_input")
    
    # ====== FILTER DATA =====
    filtered_df = status_df.copy()
    
    if filter_option == "Downloaded Only":
        filtered_df = filtered_df[filtered_df['downloaded']]
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
        new_selection = filtered_df['ticker'].tolist()
        # Only rerun if selection actually changed
        if set(new_selection) != set(st.session_state.dm_selected_tickers):
            num_to_select = len(new_selection)
            if num_to_select > 100:
                st.warning(f"⚠️ Selecting {num_to_select} stocks may cause the page to slow down. Consider using filters to narrow your selection.")
            st.session_state.dm_selected_tickers = new_selection
            st.rerun()
    
    if bulk_missing_clicked:
        missing_stocks = filtered_df[~filtered_df['downloaded']]['ticker'].tolist()
        # Only rerun if selection actually changed
        if set(missing_stocks) != set(st.session_state.dm_selected_tickers):
            num_to_select = len(missing_stocks)
            if num_to_select > 100:
                st.warning(f"⚠️ Selecting {num_to_select} stocks may cause the page to slow down.")
            st.session_state.dm_selected_tickers = missing_stocks
            st.rerun()
    
    if bulk_invert_clicked:
        current_selected = set(st.session_state.dm_selected_tickers)
        all_filtered = set(filtered_df['ticker'].tolist())
        new_selection = list(all_filtered - current_selected)
        # Only rerun if selection actually changed
        if set(new_selection) != set(st.session_state.dm_selected_tickers):
            num_to_select = len(new_selection)
            if num_to_select > 100:
                st.warning(f"⚠️ Selecting {num_to_select} stocks may cause the page to slow down.")
            st.session_state.dm_selected_tickers = new_selection
            st.rerun()
    
    st.write("---")
    
    # ====== DATA TABLE =====
    st.write("### 📋 Stock Inventory")
    
    # Show filtered count and selected count on same line
    selected_count = len(st.session_state.dm_selected_tickers)
    if selected_count > 0:
        st.write(f"Showing **{len(filtered_df)}** of **{len(status_df)}** stocks | **{selected_count}** selected")
    else:
        st.write(f"Showing **{len(filtered_df)}** of **{len(status_df)}** stocks")
    
    # Interactive table with checkboxes
    display_df = filtered_df.copy()
    display_df['select'] = display_df['ticker'].isin(st.session_state.dm_selected_tickers)
    
    # Add download status column with checkmarks
    display_df['download_status'] = display_df['downloaded'].apply(lambda x: '✅' if x else '❌')
    
    # Reorder columns to include download status and validation status
    table_columns = ['select', 'ticker', 'download_status', 'validation_status', 'records', 'file_size', 'date_range', 'last_updated']
    display_df = display_df[table_columns]
    display_df.columns = ['Select', 'Ticker', '📥 Downloaded', '✅ Verified', 'Records', 'Size', 'Date Range', 'Last Updated']
    
    # Use a stable key based on filter to persist checkbox state
    editor_key = f"stock_editor_{filter_option}_{sort_option}"
    
    # Interactive table
    edited_df = st.data_editor(
        display_df,
        disabled=['Ticker', '📥 Downloaded', '✅ Verified', 'Records', 'Size', 'Date Range', 'Last Updated'],
        hide_index=True,
        use_container_width=True,
        height=400,
        key=editor_key,
        column_config={
            'Select': st.column_config.CheckboxColumn('Select', default=False, width='small'),
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            '📥 Downloaded': st.column_config.TextColumn('📥 Downloaded', width='small'),
            '✅ Verified': st.column_config.TextColumn('✅ Verified', width='small'),
            'Records': st.column_config.NumberColumn('Records', format='%d', width='small'),
            'Size': st.column_config.TextColumn('Size', width='small'),
            'Date Range': st.column_config.TextColumn('Date Range', width='medium'),
            'Last Updated': st.column_config.TextColumn('Last Updated', width='medium')
        }
    )
    
    # Update selection from table - synchronize session state with editor state
    new_selection = edited_df[edited_df['Select']]['Ticker'].tolist()
    # Always update to keep in sync, but don't force rerun
    st.session_state.dm_selected_tickers = new_selection
    
    st.write("---")
    
    # ====== ACTIONS: DOWNLOAD, VALIDATE, DELETE (No title, directly under table) =====
    selected_tickers = st.session_state.dm_selected_tickers
    selected_count = len(selected_tickers)
    selected_downloaded = [t for t in selected_tickers if t in downloaded_tickers]
    selected_not_downloaded = [t for t in selected_tickers if t not in downloaded_tickers]

    # Action buttons in tabs - reordered: Download, Validate, Delete
    tab1, tab2, tab3 = st.tabs(["📥 Download", "✅ Validate", "🗑️ Delete"])

    # TAB 1: DOWNLOAD
    with tab1:
        st.write("**Download Selected Data**")

        force_redownload = False
        if selected_count and selected_downloaded:
            force_redownload = st.checkbox(
                "Force re-download for already downloaded tickers",
                value=False,
                key="dm_force_redownload",
                help="Re-fetch data and overwrite existing CSVs for the selected tickers."
            )

        download_targets = selected_tickers if force_redownload else selected_not_downloaded
        download_disabled = len(download_targets) == 0

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

        end_date = datetime.now()
        if date_range == "1 Year":
            start_date = end_date - timedelta(days=365)
        elif date_range == "3 Years":
            start_date = end_date - timedelta(days=365 * 3)
        elif date_range == "5 Years":
            start_date = end_date - timedelta(days=365 * 5)
        elif date_range == "10 Years":
            start_date = end_date - timedelta(days=365 * 10)
        else:
            start_date = datetime(2010, 1, 1)

        download_label = f"📥 Download Now ({len(download_targets)} stocks)" if download_targets else "📥 Download Now"
        key_suffix = ",".join(sorted(download_targets[:5])) if download_targets else "none"
        mode_suffix = "force" if force_redownload else "fresh"
        download_key = f"download_btn_{mode_suffix}_{key_suffix}"

        download_clicked = st.button(
            download_label,
            type="primary",
            use_container_width=True,
            key=download_key,
            disabled=download_disabled
        )

        if download_disabled:
            if selected_count == 0:
                st.caption("Select tickers from the table above to enable downloads.")
            elif not force_redownload and not selected_not_downloaded:
                st.caption("All selected tickers already have data. Toggle force re-download to refresh files.")

        if download_clicked:
            with st.spinner(f"Downloading {len(download_targets)} stocks..."):
                try:
                    ingestion = DataIngestion(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("⏳ Downloading data...")
                    results = ingestion.download_all_tickers(download_targets, max_workers=max_workers)

                    progress_bar.progress(50)

                    successful = sum(1 for v in results.values() if v)
                    failed = len(results) - successful
                    successful_tickers = [k for k, v in results.items() if v]

                    validation_results = {}
                    if successful_tickers:
                        status_text.text("🔍 Validating downloaded data...")
                        for i, ticker in enumerate(successful_tickers):
                            try:
                                data = loader.load_ticker_ohlcv(ticker)
                                if data is not None:
                                    validation_results[ticker] = validator.validate_price_data(ticker, data)
                                    st.session_state.dm_validation_results[ticker] = validation_results[ticker]
                                else:
                                    validation_results[ticker] = {
                                        'is_valid': False,
                                        'discrepancies': ["No data available after download"]
                                    }
                                    st.session_state.dm_validation_results[ticker] = validation_results[ticker]
                                progress_bar.progress(50 + int((i + 1) / len(successful_tickers) * 50))
                            except Exception as e:
                                validation_results[ticker] = {
                                    'is_valid': False,
                                    'discrepancies': [f"Validation error: {e}"]
                                }
                                st.session_state.dm_validation_results[ticker] = validation_results[ticker]

                        save_validation_cache(st.session_state.dm_validation_results)

                    progress_bar.progress(100)
                    status_text.empty()

                    if successful_tickers:
                        valid_count = sum(1 for v in validation_results.values() if v.get('is_valid'))
                        issues = {t: r for t, r in validation_results.items() if not r.get('is_valid')}

                        if failed == 0 and not issues:
                            st.success(f"✅ Downloaded and validated all {successful} stocks successfully! All data is valid.")
                        elif failed == 0:
                            st.warning(f"⚠️ Downloaded {successful} stocks: {valid_count} valid, {len(issues)} with issues")
                        else:
                            st.warning(f"⚠️ Downloaded {successful} stocks ({valid_count} valid, {len(issues)} with issues), {failed} failed")

                        if issues:
                            with st.expander(f"📋 View Detailed Validation Results ({len(issues)} stocks with issues)", expanded=True):
                                for ticker, result in issues.items():
                                    st.markdown(f"### 🔍 {ticker}")
                                    discrepancies = result.get('discrepancies', [])
                                    if discrepancies:
                                        st.markdown("**❌ Discrepancies:**")
                                        for entry in discrepancies:
                                            st.markdown(f"- {entry}")
                                    else:
                                        st.info("No specific discrepancies reported.")

                                    avg_diff = result.get('avg_difference_pct')
                                    if avg_diff is not None:
                                        st.write(f"- Average price difference: {avg_diff:.4f}%")
                                    max_diff = result.get('max_difference_pct')
                                    if max_diff is not None:
                                        st.write(f"- Max price difference: {max_diff:.4f}%")
                                    st.markdown("---")

                        with st.expander(f"📊 View All Validation Details ({len(validation_results)} stocks)", expanded=False):
                            for ticker, result in validation_results.items():
                                is_valid = result.get('is_valid', False)
                                header_icon = "✅" if is_valid else "❌"
                                st.markdown(f"### {header_icon} {ticker}")
                                st.write(f"- Quality: {result.get('data_quality', 'UNKNOWN')}")
                                st.write(f"- Confidence: {result.get('confidence_score', 0.0):.2f}%")

                                discrepancies = result.get('discrepancies', [])
                                if discrepancies:
                                    st.write("Discrepancies:")
                                    for entry in discrepancies:
                                        st.markdown(f"  - {entry}")
                                st.markdown("---")
                    else:
                        st.warning("⚠️ Download completed but no files were saved. Please check logs.")

                    st.session_state.dm_selected_tickers = successful_tickers
                    st.session_state.dm_validation_results.update(validation_results if successful_tickers else {})

                except Exception as e:
                    st.error(f"❌ Download error: {e}")

    # TAB 2: VALIDATE
    with tab2:
        st.write("**Validate Downloaded Data**")
        validate_disabled = len(selected_downloaded) == 0
        validator_workers = st.number_input(
            "Validation Threads",
            min_value=1,
            max_value=10,
            value=3,
            key="validation_workers",
            disabled=validate_disabled
        )
        
        validator_mode = st.radio(
            "Validation Depth",
            ["Quick (checks key metrics)", "Deep (full price comparison)"],
            index=0,
            key="validation_mode",
            disabled=validate_disabled
        )
        
        validate_label = (
            f"✅ Validate {len(selected_downloaded)} Stock(s)"
            if selected_downloaded else "✅ Validate Selected"
        )
        validate_clicked = st.button(
            validate_label,
            type="primary",
            key="validate_btn",
            disabled=validate_disabled
        )
        
        if validate_disabled:
            if selected_count == 0:
                st.caption("Select tickers in the table to enable validation.")
            else:
                st.caption("Only downloaded tickers can be validated.")
        elif validate_clicked:
            with st.spinner(f"Validating {len(selected_downloaded)} downloads..."):
                try:
                    validation_results = {}
                    status_text = st.empty()
                    progress_bar = st.progress(0)

                    for i, ticker in enumerate(selected_downloaded):
                        status_text.text(f"🔍 Validating {ticker} ({i + 1}/{len(selected_downloaded)})")
                        try:
                            data = loader.load_ticker_ohlcv(ticker)
                            if data is None or data.empty:
                                validation_results[ticker] = {
                                    'is_valid': False,
                                    'data_quality': 'MISSING',
                                    'confidence_score': 0.0,
                                    'discrepancies': ["No data available for validation"]
                                }
                            elif validator_mode.startswith("Quick"):
                                summary = validator.quick_validate(data)
                                validation_results[ticker] = {
                                    'is_valid': summary['overall_score'] >= 0.7,
                                    'data_quality': summary['quality_level'],
                                    'confidence_score': summary['overall_score'] * 100,
                                    'discrepancies': summary.get('warnings', [])
                                }
                            else:
                                validation_results[ticker] = validator.validate_price_data(ticker, data)
                        except Exception as e:
                            validation_results[ticker] = {
                                'is_valid': False,
                                'data_quality': 'ERROR',
                                'confidence_score': 0.0,
                                'discrepancies': [f"Validation error: {e}"]
                            }

                        progress_bar.progress(int(((i + 1) / len(selected_downloaded)) * 100))

                    status_text.empty()
                    progress_bar.empty()

                    st.session_state.dm_validation_results.update(validation_results)
                    save_validation_cache(st.session_state.dm_validation_results)

                    valid_count = sum(1 for v in validation_results.values() if v.get('is_valid'))
                    issues_count = len(validation_results) - valid_count

                    if issues_count == 0:
                        st.success(f"✅ All {len(validation_results)} stocks validated successfully!")
                    else:
                        st.warning(f"⚠️ Validation complete: {valid_count} valid, {issues_count} with issues")

                    with st.expander("📋 View Validation Results", expanded=True):
                        for ticker, result in validation_results.items():
                            is_valid = result.get('is_valid', False)
                            status_icon = "✅" if is_valid else "❌"
                            st.markdown(f"### {status_icon} {ticker}")
                            st.write(f"- Quality: {result.get('data_quality', 'UNKNOWN')}")
                            st.write(f"- Confidence: {result.get('confidence_score', 0.0):.2f}%")

                            discrepancies = result.get('discrepancies', [])
                            if discrepancies:
                                st.write("Discrepancies:")
                                for entry in discrepancies:
                                    st.markdown(f"  - {entry}")
                            st.markdown("---")

                    st.info("💡 The table will refresh after validation completes.")
                    time.sleep(2)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Validation error: {e}")

    # TAB 3: DELETE
    with tab3:
        st.write("**Delete Downloaded Data**")
        delete_disabled = len(selected_downloaded) == 0
        delete_label = (
            f"🗑️ Delete {len(selected_downloaded)} File(s)"
            if selected_downloaded else "🗑️ Delete Selected"
        )
        delete_clicked = st.button(
            delete_label,
            type="secondary",
            use_container_width=True,
            key="delete_btn",
            disabled=delete_disabled
        )
        
        if delete_disabled:
            if selected_count == 0:
                st.caption("Select tickers in the table to enable deletion.")
            else:
                st.caption("Only downloaded tickers can be deleted.")
        else:
            st.warning("⚠️ This will permanently delete the CSV files!")
            if delete_clicked:
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

    if selected_count == 0:
        st.info("💡 **Tip:** Select stocks using checkboxes or bulk actions to download/manage data.")




if __name__ == "__main__":
    show()

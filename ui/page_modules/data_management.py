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
        
        # Define available timeframes
        TIMEFRAMES = ['15m', '30m', '1h', '4h', '12h', '1d', '1w', '1M']
        
        # Get all S&P 500 tickers
        ingestion = DataIngestion()
        all_sp500_tickers = ingestion.get_sp500_tickers()
        
        # Get downloaded tickers for each timeframe
        available_by_timeframe = loader.get_available_tickers_by_timeframe()
        
        # Build status table with validation and timeframe columns
        ticker_status = []
        for ticker in all_sp500_tickers:
            status = {
                'ticker': ticker,
                'downloaded': False,  # Will be True if ANY timeframe is downloaded
            }
            
            # Add columns for each timeframe
            for tf in TIMEFRAMES:
                status[f'tf_{tf}'] = '✅' if ticker in available_by_timeframe.get(tf, []) else '❌'
                if ticker in available_by_timeframe.get(tf, []):
                    status['downloaded'] = True
            
            ticker_status.append(status)
        
        status_df = pd.DataFrame(ticker_status)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # ====== OVERVIEW (No title) =====
    col1, col2, col3 = st.columns(3)
    
    downloaded_count = status_df['downloaded'].sum()
    # Count total downloaded files across all timeframes
    total_files = sum(1 for tf in TIMEFRAMES for ticker_list in [available_by_timeframe.get(tf, [])] for _ in ticker_list)
    universe_size = len(status_df)
    
    with col1:
        st.metric("📥 Downloaded", f"{downloaded_count}/{universe_size}")
    with col2:
        st.metric("� Total Files", f"{total_files}")
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
            ["Ticker (A-Z)", "Ticker (Z-A)"],
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
    # ALWAYS use session state as source of truth for checkbox state
    display_df['select'] = display_df['ticker'].isin(st.session_state.dm_selected_tickers)
    
    # Reorder columns to include timeframe status
    table_columns = ['select', 'ticker', 'tf_1d', 'tf_1h', 'tf_15m', 'tf_30m', 'tf_4h', 'tf_1w', 'tf_1M']
    display_df = display_df[table_columns]
    display_df.columns = ['Select', 'Ticker', '1d', '1h', '15m', '30m', '4h', '1w', '1M']
    
    # Use a completely stable key - don't include filter/sort to maintain state
    editor_key = "stock_editor_main"
    
    # Column configurations for timeframe columns
    timeframe_col_config = {}
    for tf in ['1d', '1h', '15m', '30m', '4h', '1w', '1M']:
        timeframe_col_config[tf] = st.column_config.TextColumn(tf, width='small')
    
    # Interactive table
    edited_df = st.data_editor(
        display_df,
        disabled=['Ticker', '1d', '1h', '15m', '30m', '4h', '1w', '1M'],
        hide_index=True,
        use_container_width=True,
        height=400,
        key=editor_key,
        column_config={
            'Select': st.column_config.CheckboxColumn('Select', default=False, width='small'),
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            **timeframe_col_config
        }
    )
    
    # Update selection from table - synchronize session state with editor state
    new_selection = edited_df[edited_df['Select']]['Ticker'].tolist()
    # Update session state with the new selection from the editor
    # This captures user clicks on the checkboxes in the table
    st.session_state.dm_selected_tickers = new_selection
    
    st.write("---")
    
    # ====== ACTIONS: DOWNLOAD, VALIDATE, DELETE (No title, directly under table) =====
    # Use session state for selected tickers (persists across reruns from checkbox clicks)
    selected_tickers = st.session_state.dm_selected_tickers
    selected_count = len(selected_tickers)
    
    # Get list of downloaded tickers (checking 1d timeframe as reference)
    downloaded_tickers_1d = available_by_timeframe.get('1d', [])
    selected_downloaded = [t for t in selected_tickers if t in downloaded_tickers_1d]

    # Action buttons in tabs - reordered: Download, Validate, Delete
    tab1, tab2, tab3 = st.tabs(["📥 Download", "✅ Validate", "🗑️ Delete"])

    # TAB 1: DOWNLOAD
    with tab1:
        st.write("**Download Selected Data**")
        
        # Timeframe selector
        st.write("**Select Timeframes to Download:**")
        tf_cols = st.columns(4)
        download_timeframes = []
        with tf_cols[0]:
            if st.checkbox("📅 1 Day", value=True, key="tf_1d"):
                download_timeframes.append('1d')
            if st.checkbox("⏱️ 1 Hour", value=False, key="tf_1h"):
                download_timeframes.append('1h')
        with tf_cols[1]:
            if st.checkbox("⏱️ 15 Min", value=False, key="tf_15m"):
                download_timeframes.append('15m')
            if st.checkbox("⏱️ 30 Min", value=False, key="tf_30m"):
                download_timeframes.append('30m')
        with tf_cols[2]:
            if st.checkbox("⏱️ 4 Hour", value=False, key="tf_4h"):
                download_timeframes.append('4h')
            if st.checkbox("📊 1 Week", value=False, key="tf_1w"):
                download_timeframes.append('1w')
        with tf_cols[3]:
            if st.checkbox("📊 1 Month", value=False, key="tf_1M"):
                download_timeframes.append('1M')
        
        # Calculate download jobs
        download_job_count = len(selected_tickers) * len(download_timeframes)
        
        # Check which tickers need downloading for selected timeframes
        missing_combos = []
        for ticker in selected_tickers:
            for tf in download_timeframes:
                if ticker not in available_by_timeframe.get(tf, []):
                    missing_combos.append((ticker, tf))

        force_redownload = st.checkbox(
            "Force re-download existing data",
            value=False,
            key="dm_force_redownload",
            help="Re-fetch data and overwrite existing CSVs for the selected tickers and timeframes."
        )
        
        download_disabled = (selected_count == 0 or len(download_timeframes) == 0)

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

        download_label = f"📥 Download Now ({download_job_count} jobs)" if download_job_count else "📥 Download Now"
        
        # Initialize download trigger flag
        if 'dm_download_triggered' not in st.session_state:
            st.session_state.dm_download_triggered = False
            
        # Callback to trigger download
        def trigger_download():
            st.session_state.dm_download_triggered = True
            st.session_state.dm_download_params = {
                'selected_tickers': selected_tickers.copy(),
                'download_timeframes': download_timeframes.copy(),
                'start_date': start_date,
                'end_date': end_date,
                'max_workers': max_workers,
                'force_redownload': force_redownload
            }

        download_clicked = st.button(
            download_label,
            type="primary",
            use_container_width=True,
            key="dm_download_btn_stable",
            disabled=download_disabled,
            on_click=trigger_download
        )

        if download_disabled:
            if selected_count == 0:
                st.caption("Select tickers from the table above to enable downloads.")
            elif len(download_timeframes) == 0:
                st.caption("Select at least one timeframe to download.")
        elif not force_redownload and not missing_combos:
            st.info("💡 All selected tickers already have data for the chosen timeframe(s). Toggle 'Force re-download' to refresh files.")

        # Check if download was triggered
        if st.session_state.dm_download_triggered:
            st.session_state.dm_download_triggered = False  # Reset flag
            
            # Get stored parameters
            params = st.session_state.get('dm_download_params', {})
            selected_tickers = params.get('selected_tickers', [])
            download_timeframes = params.get('download_timeframes', [])
            start_date = params.get('start_date')
            end_date = params.get('end_date')
            max_workers = params.get('max_workers', 5)
            force_redownload = params.get('force_redownload', False)
            
            download_job_count = len(selected_tickers) * len(download_timeframes)
            
            with st.spinner(f"Downloading {download_job_count} job(s) across {len(download_timeframes)} timeframe(s)..."):
                try:
                    ingestion = DataIngestion(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        timeframes=download_timeframes
                    )

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("⏳ Downloading data...")
                    results = ingestion.download_all_tickers(
                        tickers=selected_tickers,
                        timeframes=download_timeframes,
                        max_workers=max_workers,
                        skip_existing=not force_redownload
                    )

                    progress_bar.progress(100)
                    status_text.empty()

                    # Count successful and failed jobs across all timeframes
                    total_jobs = sum(len(tf_results) for tf_results in results.values())
                    successful_jobs = sum(1 for tf_results in results.values() 
                                        for status in tf_results.values() if status == 'downloaded')
                    skipped_jobs = sum(1 for tf_results in results.values() 
                                      for status in tf_results.values() if status == 'skipped')
                    failed_jobs = total_jobs - successful_jobs - skipped_jobs
                    
                    # Store results in session state for display at the bottom
                    st.session_state.dm_last_download_results = {
                        'results': results,
                        'total_jobs': total_jobs,
                        'successful_jobs': successful_jobs,
                        'skipped_jobs': skipped_jobs,
                        'failed_jobs': failed_jobs,
                        'timeframes': download_timeframes
                    }
                    
                    # Trigger rerun to refresh the table with checkmarks
                    st.rerun()

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

    # Display download results at the bottom if available
    if 'dm_last_download_results' in st.session_state and st.session_state.dm_last_download_results:
        st.markdown("---")
        st.markdown("### 📊 Last Download Results")
        
        results_data = st.session_state.dm_last_download_results
        results = results_data.get('results', {})
        total_jobs = results_data.get('total_jobs', 0)
        successful_jobs = results_data.get('successful_jobs', 0)
        skipped_jobs = results_data.get('skipped_jobs', 0)
        failed_jobs = results_data.get('failed_jobs', 0)
        timeframes = results_data.get('timeframes', [])
        
        # Summary
        if failed_jobs == 0 and skipped_jobs == 0:
            st.success(f"✅ Successfully downloaded all {successful_jobs} job(s) across {len(timeframes)} timeframe(s)!")
        elif failed_jobs == 0:
            st.success(f"✅ Downloaded {successful_jobs} job(s), skipped {skipped_jobs} existing")
        else:
            st.warning(f"⚠️ Downloaded {successful_jobs} job(s), skipped {skipped_jobs}, failed {failed_jobs}")
        
        # Show breakdown by timeframe
        with st.expander(f"📋 View Details by Timeframe ({len(timeframes)} timeframe(s))", expanded=True):
            for tf in timeframes:
                tf_results = results.get(tf, {})
                tf_downloaded = sum(1 for status in tf_results.values() if status == 'downloaded')
                tf_skipped = sum(1 for status in tf_results.values() if status == 'skipped')
                tf_failed = sum(1 for status in tf_results.values() if status == 'failed')
                
                st.markdown(f"#### ⏱️ {tf.upper()}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Downloaded", tf_downloaded)
                with col2:
                    st.metric("➖ Skipped", tf_skipped)
                with col3:
                    st.metric("❌ Failed", tf_failed)
                
                # Show failed tickers if any
                if tf_failed > 0:
                    failed_tickers = [ticker for ticker, status in tf_results.items() if status == 'failed']
                    st.warning(f"Failed tickers: {', '.join(failed_tickers)}")
                
                st.markdown("---")
        
        # Add button to clear results
        if st.button("🔄 Clear Results", key="clear_download_results"):
            st.session_state.dm_last_download_results = None
            st.rerun()

    if selected_count == 0:
        st.info("💡 **Tip:** Select stocks using checkboxes or bulk actions to download/manage data.")




if __name__ == "__main__":
    show()

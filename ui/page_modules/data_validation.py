"""
Data Validation Page
Extracted from main dashboard for modular architecture.
"""

"""
Streamlit Dashboard for QuantSearch Trading Platform
Interactive UI for strategy backtesting, visualization, and analysis.
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
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

def summarize_validation(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute headline metrics for validation results.
    
    Args:
        results_df: DataFrame returned from validation runs
    
    Returns:
        Dictionary with total tickers, valid count, rate (%), and average confidence
    """
    if results_df.empty:
        return {
            'total_tickers': 0,
            'valid_tickers': 0,
            'validation_rate': 0.0,
            'avg_confidence': 0.0
        }
    
    total = len(results_df)
    valid = int(results_df['is_valid'].sum()) if 'is_valid' in results_df else 0
    validation_rate = (valid / total * 100) if total else 0.0
    avg_confidence = float(results_df['confidence_score'].mean()) if 'confidence_score' in results_df else 0.0
    
    return {
        'total_tickers': total,
        'valid_tickers': valid,
        'validation_rate': validation_rate,
        'avg_confidence': avg_confidence
    }


# Note: DataLoader is NOT cached - always create fresh instance
# to avoid stale data issues


# Rename function to show() for consistency
def show():
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
        
        summary = summarize_validation(validation_df)
        
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
        summary = summarize_validation(validation_df)
        
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
    st.markdown('<div class="sub-header">🔍 Multi-Source Validation (Advanced)</div>', unsafe_allow_html=True)
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


if __name__ == "__main__":
    show()

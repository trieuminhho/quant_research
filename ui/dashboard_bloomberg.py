"""
QuantSearch Terminal - Bloomberg-Inspired Streamlit Dashboard
ML-Driven Systematic Trading Research Platform
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
    page_title="QuantSearch Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

styles.apply_dark_theme()

# Bloomberg-inspired CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    :root {
        --bg-primary: var(--bg);
        --bg-panel: var(--bg-2);
        --bg-card: var(--card);
        --text-primary: var(--text);
        --text-muted: var(--muted);
        --accent-cyan: var(--accent-2);
        --accent-amber: var(--warn);
        --positive: var(--success);
        --negative: var(--danger);
        --border: var(--border);
    }
    
    /* Main Container */
    .stApp {
        background-color: var(--bg-primary);
    }
    
    /* Headers */
    .terminal-header {
        font-family: 'Inter Tight', -apple-system, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--accent-cyan);
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    
    .section-header {
        font-family: 'Inter Tight', -apple-system, sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--accent-cyan);
        padding-bottom: 0.5rem;
    }
    
    .subsection-header {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-amber);
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-card) 100%);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        border-color: var(--accent-cyan);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 188, 212, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--accent-cyan);
    }
    
    .metric-label {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-cyan);
        line-height: 1;
    }
    
    .metric-value.positive {
        color: var(--positive);
    }
    
    .metric-value.negative {
        color: var(--negative);
    }
    
    .metric-change {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        color: var(--text-muted);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-excellent {
        background-color: rgba(0, 200, 83, 0.15);
        color: var(--positive);
        border: 1px solid var(--positive);
    }
    
    .status-good {
        background-color: rgba(0, 188, 212, 0.15);
        color: var(--accent-cyan);
        border: 1px solid var(--accent-cyan);
    }
    
    .status-warning {
        background-color: rgba(255, 179, 0, 0.15);
        color: var(--accent-amber);
        border: 1px solid var(--accent-amber);
    }
    
    .status-error {
        background-color: rgba(244, 67, 54, 0.15);
        color: var(--negative);
        border: 1px solid var(--negative);
    }
    
    /* Data Tables */
    .dataframe {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.875rem;
        background-color: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    
    .dataframe th {
        background-color: var(--bg-card) !important;
        color: var(--accent-cyan) !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
    }
    
    .dataframe td {
        padding: 0.75rem !important;
        border-bottom: 1px solid var(--border) !important;
    }
    
    .dataframe tr:hover {
        background-color: rgba(0, 188, 212, 0.05) !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: var(--bg-panel);
        border-right: 1px solid var(--border);
    }
    
    .css-1d391kg .stRadio label {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--text-primary);
        font-weight: 500;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    
    .css-1d391kg .stRadio label:hover {
        background-color: rgba(0, 188, 212, 0.1);
        color: var(--accent-cyan);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, rgba(34,211,238,0.45) 100%);
        color: var(--bg-primary);
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 188, 212, 0.3);
    }
    
    /* Input Fields */
    .stTextInput input, .stSelectbox select, .stMultiSelect select {
        background-color: var(--bg-panel);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 6px;
        font-family: 'Roboto Mono', monospace;
        padding: 0.75rem;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 600;
        color: var(--text-muted);
        padding: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan);
        border-bottom: 3px solid var(--accent-cyan);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--bg-panel);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 600;
        border: 1px solid var(--border);
        border-radius: 6px;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--accent-cyan);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: rgba(0, 200, 83, 0.1);
        border-left: 4px solid var(--positive);
        color: var(--positive);
    }
    
    .stError {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid var(--negative);
        color: var(--negative);
    }
    
    .stWarning {
        background-color: rgba(255, 179, 0, 0.1);
        border-left: 4px solid var(--accent-amber);
        color: var(--accent-amber);
    }
    
    .stInfo {
        background-color: rgba(0, 188, 212, 0.1);
        border-left: 4px solid var(--accent-cyan);
        color: var(--accent-cyan);
    }
    
    /* Charts */
    .plotly-graph-div {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Sparklines */
    .sparkline {
        height: 40px;
        margin-top: 0.5rem;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: var(--accent-cyan);
    }
    
    /* Text Colors */
    p, span, div {
        color: var(--text-primary);
    }
    
    .text-muted {
        color: var(--text-muted);
    }
    
    /* Code Blocks */
    code {
        background-color: var(--bg-card);
        color: var(--accent-cyan);
        font-family: 'Roboto Mono', monospace;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    pre {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration (cached)."""
    return get_config()


def render_metric_card(label: str, value: str, change: str = None, status: str = "neutral"):
    """Render a Bloomberg-style metric card."""
    status_class = "positive" if status == "positive" else "negative" if status == "negative" else ""
    change_html = f'<div class="metric-change">{change}</div>' if change else ""
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {status_class}">{value}</div>
        {change_html}
    </div>
    """


def render_status_badge(status: str, text: str):
    """Render a status badge."""
    status_map = {
        "excellent": "status-excellent",
        "good": "status-good",
        "warning": "status-warning",
        "error": "status-error"
    }
    status_class = status_map.get(status.lower(), "status-good")
    return f'<span class="status-badge {status_class}">{text}</span>'


def create_sparkline(data: pd.Series, color: str = "#22D3EE"):
    """Create a mini sparkline chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data.values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba(0, 188, 212, 0.1)',
        hoverinfo='skip'
    ))
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def show_home_page():
    """Bloomberg-inspired home/dashboard page."""
    st.markdown('<div class="terminal-header">⚡ QUANTSEARCH TERMINAL</div>', unsafe_allow_html=True)
    st.markdown('<p class="text-muted" style="font-size: 1.1rem; margin-bottom: 2rem;">ML-Driven Systematic Trading Research Platform</p>', unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown('<div class="section-header">SYSTEM OVERVIEW</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(render_metric_card("STRATEGIES", "12", "↑ 2 new", "positive"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card("BACKTESTS", "45", "↑ 8 today", "positive"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("AVG SHARPE", "2.34", "↑ 0.12", "positive"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_metric_card("WIN RATE", "67%", "↑ 3%", "positive"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance Metrics
    st.markdown('<div class="section-header">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(render_metric_card("CAGR", "18.5%", "vs 15.2% benchmark", "positive"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card("MAX DRAWDOWN", "-12.3%", "within tolerance", "neutral"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("SORTINO RATIO", "3.21", "↑ 0.45", "positive"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Health Status
    st.markdown('<div class="section-header">DATA HEALTH STATUS</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="subsection-header">Raw Data</div>', unsafe_allow_html=True)
        st.markdown(render_status_badge("excellent", "SYNCED") + " 500 tickers loaded", unsafe_allow_html=True)
        st.markdown(render_status_badge("good", "FRESH") + " Updated 2 hours ago", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Features</div>', unsafe_allow_html=True)
        st.markdown(render_status_badge("excellent", "COMPLETE") + " All features computed", unsafe_allow_html=True)
        st.markdown(render_status_badge("good", "VALIDATED") + " No missing data", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="subsection-header">Predictions</div>', unsafe_allow_html=True)
        st.markdown(render_status_badge("excellent", "READY") + " Models trained", unsafe_allow_html=True)
        st.markdown(render_status_badge("good", "CURRENT") + " Latest signals available", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown('<div class="section-header">RECENT ACTIVITY</div>', unsafe_allow_html=True)
    
    activity_data = {
        "Time": ["10:42 AM", "09:15 AM", "Yesterday", "Yesterday", "2 days ago"],
        "Event": ["Backtest Completed", "Features Updated", "Model Training", "Data Sync", "Strategy Created"],
        "Status": ["✅ Success", "✅ Success", "✅ Success", "✅ Success", "✅ Success"],
        "Details": ["ML_Momentum_v2", "Technical Indicators", "LightGBM Ensemble", "500 tickers", "Mean Reversion"]
    }
    df = pd.DataFrame(activity_data)
    st.dataframe(df, width="stretch", hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown('<div class="section-header">QUICK ACTIONS</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 RUN BACKTEST", width="stretch"):
            st.info("Navigate to Strategy Backtesting page using the sidebar")
    with col2:
        if st.button("🔄 REFRESH DATA", width="stretch"):
            st.info("Navigate to Data Management page using the sidebar")
    with col3:
        if st.button("🧪 TRAIN MODEL", width="stretch"):
            st.info("Navigate to Model Training page using the sidebar")
    with col4:
        if st.button("📈 VIEW CHARTS", width="stretch"):
            st.info("Navigate to Performance Analysis page using the sidebar")


def show_data_management_page():
    """Data management page with Bloomberg styling."""
    st.markdown('<div class="terminal-header">📦 DATA MANAGEMENT</div>', unsafe_allow_html=True)
    st.markdown('<p class="text-muted" style="font-size: 1.1rem; margin-bottom: 2rem;">Manage, validate, and monitor data pipeline</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📥 DOWNLOAD", "✅ VALIDATE", "👁️ VIEW & MANAGE", "🔍 INSPECT"])
    
    with tabs[0]:
        show_data_download_tab()
    
    with tabs[1]:
        show_data_validation_tab()
    
    with tabs[2]:
        show_data_viewer_tab()
    
    with tabs[3]:
        show_data_inspect_tab()


def show_data_download_tab():
    """Data download tab."""
    st.markdown('<div class="subsection-header">Download Market Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        download_option = st.radio(
            "Download Method",
            ["Single Ticker", "Multiple Tickers", "Full S&P 500"],
            horizontal=True
        )
    
    if download_option == "Single Ticker":
        ticker = st.text_input("Ticker Symbol", "AAPL", placeholder="Enter ticker symbol")
        
        if st.button("🚀 DOWNLOAD DATA", width="stretch"):
            with st.spinner(f"Downloading {ticker}..."):
                try:
                    from data.ingestion import DataIngestion
                    ingestion = DataIngestion()
                    data = ingestion.download_ticker_data(ticker)
                    if data is not None:
                        ingestion.save_ticker_data(ticker, data)
                        st.success(f"✅ Successfully downloaded {ticker} ({len(data)} rows)")
                    else:
                        st.error(f"❌ Failed to download {ticker}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    elif download_option == "Multiple Tickers":
        tickers_input = st.text_area(
            "Ticker Symbols (comma-separated)",
            "AAPL,GOOGL,MSFT,TSLA",
            placeholder="Enter tickers separated by commas"
        )
        
        if st.button("🚀 DOWNLOAD ALL", width="stretch"):
            tickers = [t.strip().upper() for t in tickers_input.split(",")]
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                from data.ingestion import DataIngestion
                ingestion = DataIngestion()
                
                success_count = 0
                for i, ticker in enumerate(tickers):
                    status_text.text(f"Downloading {ticker}... ({i+1}/{len(tickers)})")
                    data = ingestion.download_ticker_data(ticker)
                    if data is not None:
                        ingestion.save_ticker_data(ticker, data)
                        success_count += 1
                    progress_bar.progress((i + 1) / len(tickers))
                
                st.success(f"✅ Successfully downloaded {success_count}/{len(tickers)} tickers")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    else:  # Full S&P 500
        st.info("This will download all S&P 500 constituents. This may take several minutes.")
        
        if st.button("🚀 DOWNLOAD S&P 500", width="stretch"):
            with st.spinner("Downloading S&P 500 tickers..."):
                try:
                    from data.ingestion import DataIngestion
                    ingestion = DataIngestion()
                    ingestion.download_all_tickers()
                    st.success("✅ Successfully downloaded S&P 500 data")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def show_data_validation_tab():
    """Data validation tab."""
    st.markdown('<div class="subsection-header">Multi-Source Data Validation</div>', unsafe_allow_html=True)
    
    ticker = st.text_input("Ticker to Validate", "AAPL", key="val_ticker")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        use_yfinance = st.checkbox("yfinance", value=True)
    with col2:
        use_alphavantage = st.checkbox("Alpha Vantage", value=True)
    with col3:
        use_polygon = st.checkbox("Polygon.io", value=True)
    
    if st.button("🔍 VALIDATE DATA", width="stretch"):
        sources = []
        if use_yfinance:
            sources.append("yfinance")
        if use_alphavantage:
            sources.append("alphavantage")
        if use_polygon:
            sources.append("polygon")
        
        if not sources:
            st.warning("⚠️ Please select at least one data source")
            return
        
        with st.spinner(f"Validating {ticker} against {len(sources)} sources..."):
            try:
                validator = DataValidator()
                report = validator.validate_ticker_multisource(ticker, sources=sources)
                
                # Display results
                st.markdown('<div class="subsection-header">Validation Results</div>', unsafe_allow_html=True)
                
                # Overall Status
                confidence = report.get('consensus_confidence', 0)
                quality = report.get('overall_quality', 'UNKNOWN')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(render_metric_card("CONFIDENCE", f"{confidence:.1f}%", "", "positive" if confidence > 80 else "neutral"), unsafe_allow_html=True)
                with col2:
                    status = "excellent" if quality == "EXCELLENT" else "good" if quality == "GOOD" else "warning"
                    st.markdown(f"<div style='text-align: center; font-size: 2rem; padding: 1.5rem;'>{render_status_badge(status, quality)}</div>", unsafe_allow_html=True)
                with col3:
                    sources_used = report.get('sources_compared', [])
                    st.markdown(render_metric_card("SOURCES", str(len(sources_used)), "", "positive"), unsafe_allow_html=True)
                
                # Source Comparison
                if 'source_comparison' in report:
                    st.markdown('<div class="subsection-header">Source Comparison</div>', unsafe_allow_html=True)
                    comparison_df = pd.DataFrame(report['source_comparison']).T
                    st.dataframe(comparison_df, width="stretch")
                
                # Validation Details
                if 'validation_checks' in report:
                    st.markdown('<div class="subsection-header">Validation Checks</div>', unsafe_allow_html=True)
                    checks_df = pd.DataFrame(report['validation_checks'])
                    st.dataframe(checks_df, width="stretch")
                
            except Exception as e:
                st.error(f"❌ Validation error: {str(e)}")


def show_data_viewer_tab():
    """Enhanced data viewer with charting."""
    st.markdown('<div class="subsection-header">View & Analyze Stock Data</div>', unsafe_allow_html=True)
    
    # Load available tickers
    config = load_config()
    ohlcv_path = config.get_path('raw_data')
    
    if not ohlcv_path.exists():
        st.warning("⚠️ No data directory found. Please download data first.")
        return
    
    csv_files = list(ohlcv_path.glob("*.csv"))
    ticker_files = [f for f in csv_files if f.stem not in ['download_summary', 'sp500_tickers']]
    
    if not ticker_files:
        st.warning("⚠️ No ticker data found. Please download data first.")
        return
    
    tickers = sorted([f.stem for f in ticker_files])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_ticker = st.selectbox("Select Ticker", tickers, index=0)
    
    with col2:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Area"])
    
    # Time period selector
    period_col1, period_col2, period_col3, period_col4 = st.columns([1,1,1,2])
    
    with period_col1:
        period = st.selectbox("Period", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Max"], index=4)
    
    with period_col2:
        show_ma20 = st.checkbox("MA20", value=True)
    
    with period_col3:
        show_ma50 = st.checkbox("MA50", value=True)
    
    # Load and process data
    try:
        loader = DataLoader()
        data = loader.load_ticker_ohlcv(selected_ticker)
        
        # Filter by period
        end_date = data.index[-1]
        if period == "1M":
            start_date = end_date - timedelta(days=30)
        elif period == "3M":
            start_date = end_date - timedelta(days=90)
        elif period == "6M":
            start_date = end_date - timedelta(days=180)
        elif period == "YTD":
            start_date = pd.Timestamp(f"{end_date.year}-01-01")
        elif period == "1Y":
            start_date = end_date - timedelta(days=365)
        elif period == "3Y":
            start_date = end_date - timedelta(days=365*3)
        elif period == "5Y":
            start_date = end_date - timedelta(days=365*5)
        else:  # Max
            start_date = data.index[0]
        
        data_filtered = data[data.index >= start_date].copy()
        
        # Calculate MAs
        if show_ma20:
            data_filtered['ma20'] = data_filtered['close'].rolling(20).mean()
        if show_ma50:
            data_filtered['ma50'] = data_filtered['close'].rolling(50).mean()
        
        # Display current price and metrics
        current_price = data_filtered['close'].iloc[-1]
        prev_price = data_filtered['close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "positive" if price_change > 0 else "negative"
            st.markdown(render_metric_card(
                "PRICE",
                f"${current_price:.2f}",
                f"{'↑' if price_change > 0 else '↓'} ${abs(price_change):.2f} ({price_change_pct:+.2f}%)",
                status
            ), unsafe_allow_html=True)
        
        with col2:
            volume_avg = data_filtered['volume'].mean()
            st.markdown(render_metric_card(
                "AVG VOLUME",
                f"{volume_avg/1e6:.1f}M",
                f"Last {period}",
                "neutral"
            ), unsafe_allow_html=True)
        
        with col3:
            high_52w = data_filtered['high'].max()
            st.markdown(render_metric_card(
                "52W HIGH",
                f"${high_52w:.2f}",
                "",
                "neutral"
            ), unsafe_allow_html=True)
        
        with col4:
            low_52w = data_filtered['low'].min()
            st.markdown(render_metric_card(
                "52W LOW",
                f"${low_52w:.2f}",
                "",
                "neutral"
            ), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create chart
        fig = go.Figure()
        
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=data_filtered.index,
                open=data_filtered['open'],
                high=data_filtered['high'],
                low=data_filtered['low'],
                close=data_filtered['close'],
                name=selected_ticker,
                increasing_line_color='#10B981',
                decreasing_line_color='#EF4444'
            ))
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=data_filtered.index,
                y=data_filtered['close'],
                mode='lines',
                name='Close',
                line=dict(color='#22D3EE', width=2)
            ))
        else:  # Area
            fig.add_trace(go.Scatter(
                x=data_filtered.index,
                y=data_filtered['close'],
                mode='lines',
                name='Close',
                fill='tozeroy',
                line=dict(color='#22D3EE', width=2),
                fillcolor='rgba(0, 188, 212, 0.1)'
            ))
        
        # Add moving averages
        if show_ma20 and 'ma20' in data_filtered.columns:
            fig.add_trace(go.Scatter(
                x=data_filtered.index,
                y=data_filtered['ma20'],
                mode='lines',
                name='MA20',
                line=dict(color='#F59E0B', width=1.5, dash='dash')
            ))
        
        if show_ma50 and 'ma50' in data_filtered.columns:
            fig.add_trace(go.Scatter(
                x=data_filtered.index,
                y=data_filtered['ma50'],
                mode='lines',
                name='MA50',
                line=dict(color='#8B5CF6', width=1.5, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{selected_ticker} - {period}",
                font=dict(size=24, family='Inter Tight', color='#E5E7EB')
            ),
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=600,
            hovermode='x unified',
            paper_bgcolor='#0B0F1A',
            plot_bgcolor='#0F172A',
            font=dict(family='Inter', color='#E5E7EB'),
            xaxis=dict(
                gridcolor='#1F2937',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='#1F2937',
                showgrid=True
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Volume chart
        st.markdown('<div class="subsection-header">Trading Volume</div>', unsafe_allow_html=True)
        
        fig_vol = go.Figure()
        colors = ['#10B981' if data_filtered['close'].iloc[i] >= data_filtered['open'].iloc[i] else '#EF4444' 
                  for i in range(len(data_filtered))]
        
        fig_vol.add_trace(go.Bar(
            x=data_filtered.index,
            y=data_filtered['volume'],
            marker_color=colors,
            name='Volume'
        ))
        
        fig_vol.update_layout(
            template="plotly_dark",
            height=200,
            paper_bgcolor='#0B0F1A',
            plot_bgcolor='#0F172A',
            font=dict(family='Inter', color='#E5E7EB'),
            xaxis=dict(gridcolor='#1F2937'),
            yaxis=dict(gridcolor='#1F2937'),
            showlegend=False,
            margin=dict(t=0, b=0)
        )
        
        st.plotly_chart(fig_vol, width="stretch")
        
        # Data table
        with st.expander("📊 VIEW RAW DATA"):
            st.dataframe(data_filtered.tail(50), width="stretch")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")


def show_data_inspect_tab():
    """Data inspection and statistics."""
    st.markdown('<div class="subsection-header">Data Quality Inspector</div>', unsafe_allow_html=True)
    
    config = load_config()
    ohlcv_path = config.get_path('raw_data')
    
    if not ohlcv_path.exists():
        st.warning("⚠️ No data directory found.")
        return
    
    csv_files = list(ohlcv_path.glob("*.csv"))
    ticker_files = [f for f in csv_files if f.stem not in ['download_summary', 'sp500_tickers']]
    
    st.markdown(render_metric_card("TOTAL TICKERS", str(len(ticker_files)), "", "positive"), unsafe_allow_html=True)
    
    if st.button("🔍 SCAN ALL DATA", width="stretch"):
        with st.spinner("Scanning data files..."):
            results = []
            
            for ticker_file in ticker_files:
                try:
                    df = pd.read_csv(ticker_file, parse_dates=['date'], index_col='date')
                    
                    results.append({
                        'Ticker': ticker_file.stem,
                        'Rows': len(df),
                        'Start Date': df.index[0].strftime('%Y-%m-%d'),
                        'End Date': df.index[-1].strftime('%Y-%m-%d'),
                        'Missing Values': df.isnull().sum().sum(),
                        'Status': '✅' if df.isnull().sum().sum() == 0 else '⚠️'
                    })
                except Exception as e:
                    results.append({
                        'Ticker': ticker_file.stem,
                        'Rows': 0,
                        'Start Date': 'N/A',
                        'End Date': 'N/A',
                        'Missing Values': 'Error',
                        'Status': '❌'
                    })
            
            results_df = pd.DataFrame(results)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                healthy = len(results_df[results_df['Status'] == '✅'])
                st.markdown(render_metric_card("HEALTHY", str(healthy), f"{healthy/len(results_df)*100:.1f}%", "positive"), unsafe_allow_html=True)
            with col2:
                warnings = len(results_df[results_df['Status'] == '⚠️'])
                st.markdown(render_metric_card("WARNINGS", str(warnings), f"{warnings/len(results_df)*100:.1f}%", "neutral"), unsafe_allow_html=True)
            with col3:
                errors = len(results_df[results_df['Status'] == '❌'])
                status = "negative" if errors > 0 else "positive"
                st.markdown(render_metric_card("ERRORS", str(errors), f"{errors/len(results_df)*100:.1f}%", status), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(results_df, width="stretch", hide_index=True)


def main():
    """Main application."""
    
    # Sidebar Navigation
    st.sidebar.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-2); margin-bottom: 2rem;">⚡ TERMINAL</div>', unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "NAVIGATION",
        ["Dashboard", "Data Management", "Feature Engineering", "Model Training", 
         "Strategy Backtesting", "Performance Analysis", "System Configuration"],
        label_visibility="collapsed"
    )
    
    # Route to pages
    if page == "Dashboard":
        show_home_page()
    elif page == "Data Management":
        show_data_management_page()
    else:
        st.markdown(f'<div class="terminal-header">{page}</div>', unsafe_allow_html=True)
        st.info(f"🚧 {page} page coming soon...")
        st.markdown("This page is currently under development. The Bloomberg-inspired interface will include:")
        st.markdown("- High-density data displays")
        st.markdown("- Real-time metrics and updates")
        st.markdown("- Interactive charts and tables")
        st.markdown("- Cursor-first navigation")


if __name__ == "__main__":
    main()

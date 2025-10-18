"""
Stock Viewer Page
Extracted from main dashboard for modular architecture.
"""

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


# Rename function to show() for consistency
def show():
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
        dividend_yield_pct = None
        dividend_rate = info.get('dividendRate')
        if dividend_rate and current_price:
            dividend_yield_pct = (dividend_rate / current_price) * 100

        if (dividend_yield_pct is None or dividend_yield_pct <= 0) and info.get('trailingAnnualDividendYield'):
            dividend_yield_pct = info['trailingAnnualDividendYield'] * 100

        if (dividend_yield_pct is None or dividend_yield_pct <= 0) and info.get('dividendYield'):
            raw_yield = info['dividendYield']
            if raw_yield > 10:
                dividend_yield_pct = raw_yield
            elif raw_yield > 1:
                dividend_yield_pct = raw_yield
            else:
                dividend_yield_pct = raw_yield * 100

        if (dividend_yield_pct is None or dividend_yield_pct <= 0) and info.get('fiveYearAvgDividendYield'):
            avg_yield = info['fiveYearAvgDividendYield']
            dividend_yield_pct = avg_yield if avg_yield > 1 else avg_yield

        if dividend_yield_pct and dividend_yield_pct > 0:
            st.metric("Div Yield", f"{dividend_yield_pct:.2f}%")
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



if __name__ == "__main__":
    show()

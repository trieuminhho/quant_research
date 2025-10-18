"""
Performance Analysis Page
Analyze strategy performance, returns, risk, and metrics.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import get_config
from ui.components.common import render_page_header
from ui import styles

styles.apply_dark_theme()


def show():
    """Performance analysis page."""
    
    render_page_header("📉 Performance Analysis", "Analyze returns, risk, and comprehensive metrics")
    
    st.info("🚧 **Coming Soon:** Performance analysis with multi-strategy comparison")
    
    st.write("### Planned Features:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Return Metrics:**
        - Total return
        - CAGR
        - Monthly returns
        - Rolling returns
        - Benchmark comparison
        """)
    
    with col2:
        st.markdown("""
        **Risk Metrics:**
        - Volatility
        - Max drawdown
        - Sharpe ratio
        - Sortino ratio
        - Calmar ratio
        """)
    
    with col3:
        st.markdown("""
        **Visualizations:**
        - Equity curves
        - Drawdown chart
        - Monthly heatmap
        - Return distribution
        - Rolling metrics
        """)
    
    st.markdown("---")
    st.write("### Example Analysis")
    
    # Show example metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CAGR", "N/A", help="Compound Annual Growth Rate")
    
    with col2:
        st.metric("Max Drawdown", "N/A", help="Maximum peak-to-trough decline")
    
    with col3:
        st.metric("Sharpe Ratio", "N/A", help="Risk-adjusted return")
    
    with col4:
        st.metric("Sortino Ratio", "N/A", help="Downside risk-adjusted return")
    
    st.info("💡 Run a backtest in the Strategy Backtesting page to see performance analysis")


if __name__ == "__main__":
    show()

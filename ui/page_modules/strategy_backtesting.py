"""
Strategy Backtesting Page
Backtest trading strategies with T+1 execution.
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
    """Strategy backtesting page."""
    
    render_page_header("📈 Strategy Backtesting", "Backtest strategies with T+1 execution and realistic costs")
    
    st.info("🚧 **Coming Soon:** Strategy backtesting interface with realistic execution")
    
    st.write("### Planned Features:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Strategy Configuration:**
        - ML-based strategy
        - Signal generation at T
        - Execution at T+1
        - Portfolio construction
        - Max positions limit
        - Rebalance frequency
        """)
    
    with col2:
        st.markdown("""
        **Backtest Simulation:**
        - Realistic slippage
        - Transaction costs
        - Market impact
        - Equity curve
        - Trade log
        - Performance metrics
        """)
    
    st.markdown("---")
    st.write("### Configuration Preview")
    
    try:
        config = get_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strategy Settings:**")
            st.code(f"Max Positions: {config.max_positions}")
            st.code(f"Rebalance: {config.rebalance_frequency}")
            st.code(f"Initial Capital: ${config.initial_capital:,}")
        
        with col2:
            st.write("**Cost Settings:**")
            st.code(f"Commission: {config.commission_pct:.3%}")
            st.code(f"Slippage: {config.slippage_bps} bps")
            st.code(f"Execution: T+1")
    
    except Exception as e:
        st.error(f"Error loading configuration: {e}")


if __name__ == "__main__":
    show()

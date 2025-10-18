"""
Home Page
Extracted from main dashboard for modular architecture.
"""

"""
Streamlit Dashboard for QuantSearch Trading Platform
Interactive UI for strategy backtesting, visualization, and analysis.
"""

import sys
import logging
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
from features.base_feature import FeatureRegistry
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

# Note: DataLoader is NOT cached - always create fresh instance
# to avoid stale data issues


# Rename function to show() for consistency
def show():
    """Show home page with system overview."""
    
    st.markdown("<h2 class='title-glow'>Welcome to QuantSearch</h2>", unsafe_allow_html=True)
    st.write("A modular, CSV-first trading strategy research platform for systematic ML-based strategy development.")
    
    st.markdown("---")
    
    # System status
    st.markdown('<div class="sub-header title-glow">System Status</div>', unsafe_allow_html=True)
    
    try:
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


if __name__ == "__main__":
    show()

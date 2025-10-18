"""
Streamlit Dashboard for QuantSearch Trading Platform
Modular architecture with individual page modules.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all page modules
from ui.page_modules import (
    home,
    data_management,
    stock_viewer,
    data_validation,
    feature_engineering,
    model_training,
    strategy_backtesting,
    performance_analysis
)
from ui import styles

# Page configuration
st.set_page_config(
    page_title="QuantSearch - Trading Strategy Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

styles.apply_dark_theme()


def main():
    """Main dashboard application with modular pages."""
    
    st.markdown("<h1 class='title-glow'>📈 QuantSearch Trading Platform</h1>", unsafe_allow_html=True)
    st.markdown("**Production-grade ML-powered strategy research and backtesting**")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Workflow")
    
    page = st.sidebar.radio(
        "Select Page",
        [
            "Home",
            "Data Download & Validation",
            "Stock Data Viewer",
            "Feature Engineering",
            "Model Training",
            "Strategy Backtesting",
            "Performance Analysis"
        ],
        label_visibility="collapsed"
    )
    
    # Route to appropriate page
    page_mapping = {
        "Home": home,
        "Data Download & Validation": data_management,
        "Stock Data Viewer": stock_viewer,
        "Feature Engineering": feature_engineering,
        "Model Training": model_training,
        "Strategy Backtesting": strategy_backtesting,
        "Performance Analysis": performance_analysis
    }
    
    # Show the selected page
    page_module = page_mapping.get(page)
    if page_module:
        page_module.show()
    else:
        st.error(f"Page '{page}' not found")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: var(--muted); font-size: 0.75rem;">
        QuantSearch Platform v2.0<br>
        ML-Powered Trading Research
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

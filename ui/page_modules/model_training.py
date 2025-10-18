"""
Model Training Page
Train ML models with walk-forward validation.
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
    """Model training page."""
    
    render_page_header("🤖 Model Training", "Train ML models with walk-forward validation")
    
    st.info("🚧 **Coming Soon:** Model training interface with walk-forward validation")
    
    st.write("### Planned Features:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Training Configuration:**
        - LightGBM model (default)
        - Walk-forward validation
        - Expanding window approach
        - Custom hyperparameters
        - Feature importance analysis
        """)
    
    with col2:
        st.markdown("""
        **Model Evaluation:**
        - Out-of-sample performance
        - Feature importance plots
        - Prediction vs actual charts
        - Model diagnostics
        - Save/load trained models
        """)
    
    st.markdown("---")
    st.write("### Configuration Preview")
    
    try:
        config = get_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Settings:**")
            st.code(f"Model Type: {config.model_type}")
            st.code(f"Walk-Forward: Enabled")
            st.code(f"Train Window: Expanding")
        
        with col2:
            st.write("**Model Parameters:**")
            params = config.model_params
            for key, value in params.items():
                st.code(f"{key}: {value}")
    
    except Exception as e:
        st.error(f"Error loading configuration: {e}")


if __name__ == "__main__":
    show()

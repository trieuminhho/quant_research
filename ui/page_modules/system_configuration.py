"""
System Configuration Page
Extracted from main dashboard for modular architecture.
"""

"""
Streamlit Dashboard for QuantSearch Trading Platform
Interactive UI for strategy backtesting, visualization, and analysis.
"""

import sys
import time
import logging
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import numpy as np
import pandas as pd
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
    """System configuration page."""
    
    st.markdown('<div class="sub-header title-glow">System Configuration</div>', unsafe_allow_html=True)
    
    try:
        config = load_config()
        
        st.write("### Current Configuration")
        
        # Show key settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Settings:**")
            st.code(f"Start Date: {config.data_start_date}")
            st.code(f"Data Format: CSV")
            
            st.write("**Model Settings:**")
            st.code(f"Default Model: {config.model_type}")
            st.code(f"Max Positions: {config.max_positions}")
        
        with col2:
            st.write("**Backtest Settings:**")
            st.code(f"Initial Capital: ${config.initial_capital:,}")
            st.code(f"Commission: {config.commission_pct:.3%}")
            st.code(f"Slippage: {config.slippage_bps} bps")
            
            st.write("**Paths:**")
            st.code(f"Data Root: {config.get_path('raw_data')}")
        
        st.write("---")
        st.info("Edit `config.yaml` to modify system settings")
        
    except Exception as e:
        st.error(f"Error loading configuration: {e}")


if __name__ == "__main__":
    show()

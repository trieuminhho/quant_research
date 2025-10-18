"""
Feature Engineering Page
Create and manage ML features from price data.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import get_config
from utils.data_loader import DataLoader
from features.base_feature import FeatureEngine, FeatureRegistry
from ui.components.common import render_page_header
from ui import styles

styles.apply_dark_theme()


def show():
    """Feature engineering page."""
    
    render_page_header("🔧 Feature Engineering", "Create ML features from price data")
    
    try:
        # Get available features
        available_features = FeatureRegistry.list_features()
        
        st.write("### 📊 Available Feature Types")
        st.write(f"**{len(available_features)} feature types** registered in the system:")
        
        # Display features in columns
        cols = st.columns(3)
        for i, feature_name in enumerate(sorted(available_features)):
            with cols[i % 3]:
                st.info(f"✓ {feature_name}")
        
        st.markdown("---")
        
        # Feature selection
        st.write("### ⚙️ Configure Features")
        
        selected_features = st.multiselect(
            "Select features to compute",
            options=sorted(available_features),
            default=[]
        )
        
        # Ticker selection
        loader = DataLoader()
        available_tickers = loader.get_available_tickers()
        
        st.write(f"**{len(available_tickers)} tickers** available with downloaded data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker_mode = st.radio(
                "Select tickers",
                ["All downloaded", "Specific tickers"],
                horizontal=True
            )
        
        selected_tickers = []
        if ticker_mode == "Specific tickers":
            with col2:
                selected_tickers = st.multiselect(
                    "Choose tickers",
                    options=available_tickers,
                    default=[]
                )
        else:
            selected_tickers = available_tickers
        
        st.markdown("---")
        
        # Compute features
        st.write("### 🚀 Compute Features")
        
        if selected_features and selected_tickers:
            st.info(f"Ready to compute **{len(selected_features)} features** for **{len(selected_tickers)} tickers**")
            
            if st.button("▶️ Compute Features", type="primary"):
                with st.spinner(f"Computing features for {len(selected_tickers)} tickers..."):
                    try:
                        engine = FeatureEngine()
                        
                        # Add selected features
                        for feature_name in selected_features:
                            engine.add_feature(feature_name)
                        
                        # Compute features
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        success_count = 0
                        for i, ticker in enumerate(selected_tickers):
                            try:
                                status_text.text(f"Processing {ticker}... ({i+1}/{len(selected_tickers)})")
                                data = loader.load_ticker_ohlcv(ticker)
                                if data is not None:
                                    features = engine.compute_features(data)
                                    # Save features
                                    config = get_config()
                                    output_path = config.get_path('processed_features') / f"{ticker}_features.csv"
                                    features.to_csv(output_path)
                                    success_count += 1
                                progress_bar.progress(int((i + 1) / len(selected_tickers) * 100))
                            except Exception as e:
                                st.warning(f"⚠️ Error processing {ticker}: {e}")
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if success_count == len(selected_tickers):
                            st.success(f"✅ Successfully computed features for all {success_count} tickers!")
                        else:
                            st.warning(f"⚠️ Computed features for {success_count} out of {len(selected_tickers)} tickers")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        else:
            st.info("💡 Select features and tickers above to compute features")
        
        st.markdown("---")
        
        # Feature preview
        st.write("### 👁️ Feature Preview")
        
        if selected_tickers:
            preview_ticker = st.selectbox("Select ticker to preview", selected_tickers[:10])
            
            if st.button("🔍 Load Features"):
                try:
                    config = get_config()
                    feature_path = config.get_path('processed_features') / f"{preview_ticker}_features.csv"
                    
                    if feature_path.exists():
                        import pandas as pd
                        features_df = pd.read_csv(feature_path, parse_dates=['date'], index_col='date')
                        
                        st.write(f"**{len(features_df)} records, {len(features_df.columns)} features**")
                        st.dataframe(features_df.tail(20), use_container_width=True)
                        
                        # Feature statistics
                        with st.expander("📊 Feature Statistics"):
                            st.write(features_df.describe())
                    else:
                        st.warning(f"⚠️ No features found for {preview_ticker}. Compute features first.")
                        
                except Exception as e:
                    st.error(f"❌ Error loading features: {e}")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    show()

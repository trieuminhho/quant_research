"""Filters panel component."""
import streamlit as st


def render(state_key: str = "filters") -> dict:
    """Render filters panel.
    
    Args:
        state_key: Session state key for filters
        
    Returns:
        Dictionary of current filter values
    """
    # Initialize session state
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "filter_type": "All Stocks",
            "sort_by": "Ticker (A-Z)",
            "search_query": "",
            "active_filters": []
        }
    
    filters = st.session_state[state_key]
    
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    
    # Row 1: Main filters
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        filter_type = st.selectbox(
            "Filter",
            ["All Stocks", "Downloaded Only", "Not Downloaded", "Search..."],
            index=["All Stocks", "Downloaded Only", "Not Downloaded", "Search..."].index(filters["filter_type"]),
            key=f"{state_key}_filter_type"
        )
        filters["filter_type"] = filter_type
    
    with col2:
        sort_by = st.selectbox(
            "Sort By",
            ["Ticker (A-Z)", "Ticker (Z-A)", "Records (High-Low)", "Records (Low-High)", "Last Updated"],
            index=["Ticker (A-Z)", "Ticker (Z-A)", "Records (High-Low)", "Records (Low-High)", "Last Updated"].index(filters["sort_by"]),
            key=f"{state_key}_sort_by"
        )
        filters["sort_by"] = sort_by
    
    with col3:
        if filter_type == "Search...":
            search_query = st.text_input(
                "Search tickers",
                value=filters["search_query"],
                placeholder="e.g., AAPL, MSFT, GOOGL",
                key=f"{state_key}_search"
            )
            filters["search_query"] = search_query
        else:
            filters["search_query"] = ""
    
    st.write("")  # Spacing
    
    # Row 2: Bulk selection controls
    st.markdown("**Bulk Selection**")
    col1, col2, col3, col4 = st.columns(4)
    
    bulk_actions = {}
    with col1:
        bulk_actions["select_all"] = st.button("✅ Select All", key=f"{state_key}_select_all", use_container_width=True)
    with col2:
        bulk_actions["deselect_all"] = st.button("❌ Deselect All", key=f"{state_key}_deselect_all", use_container_width=True)
    with col3:
        bulk_actions["select_missing"] = st.button("📥 Select Missing", key=f"{state_key}_select_missing", use_container_width=True)
    with col4:
        bulk_actions["invert"] = st.button("🔀 Invert Selection", key=f"{state_key}_invert", use_container_width=True)
    
    filters["bulk_actions"] = bulk_actions
    
    # Advanced filters (expander)
    with st.expander("⚙️ Advanced Filters"):
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            date_range = st.date_input(
                "Date Range",
                value=[],
                key=f"{state_key}_date_range"
            )
            filters["date_range"] = date_range
        
        with adv_col2:
            min_records = st.number_input(
                "Min Records",
                min_value=0,
                value=0,
                step=100,
                key=f"{state_key}_min_records"
            )
            filters["min_records"] = min_records
        
        # Save/Load presets
        preset_col1, preset_col2 = st.columns(2)
        with preset_col1:
            if st.button("💾 Save Preset", key=f"{state_key}_save_preset"):
                st.toast("Preset saved!", icon="✅")
        with preset_col2:
            if st.button("📂 Load Preset", key=f"{state_key}_load_preset"):
                st.toast("Preset loaded!", icon="📂")
    
    # Show active filter chips
    if filters["filter_type"] != "All Stocks":
        st.markdown(f'<span class="filter-chip">{filters["filter_type"]} <span class="filter-chip-close">×</span></span>', unsafe_allow_html=True)
    
    if filters["search_query"]:
        st.markdown(f'<span class="filter-chip">Search: {filters["search_query"]} <span class="filter-chip-close">×</span></span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update session state
    st.session_state[state_key] = filters
    
    return filters

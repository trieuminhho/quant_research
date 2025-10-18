"""Inventory table component."""
import streamlit as st
import pandas as pd


def render(df: pd.DataFrame, filters: dict, state_key: str = "table") -> dict:
    """Render inventory table with selection.
    
    Args:
        df: DataFrame with columns: ticker, records, size_kb, date_start, date_end, last_updated, status
        filters: Current filter settings
        state_key: Session state key for table state
        
    Returns:
        Dictionary with selected rows
    """
    # Initialize session state
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "selected_tickers": [],
            "page": 0
        }
    
    table_state = st.session_state[state_key]
    
    # Apply filters
    filtered_df = df.copy()
    
    # KPI filter (from kpi_cards)
    kpi_filter = st.session_state.get("kpi_filter")
    if kpi_filter == "downloaded":
        filtered_df = filtered_df[filtered_df["status"] == "downloaded"]
    elif kpi_filter == "not_downloaded":
        filtered_df = filtered_df[filtered_df["status"] == "missing"]
    
    # Filter type
    if filters["filter_type"] == "Downloaded Only":
        filtered_df = filtered_df[filtered_df["status"] == "downloaded"]
    elif filters["filter_type"] == "Not Downloaded":
        filtered_df = filtered_df[filtered_df["status"] == "missing"]
    elif filters["filter_type"] == "Search..." and filters["search_query"]:
        search_terms = [term.strip().upper() for term in filters["search_query"].split(',')]
        filtered_df = filtered_df[filtered_df["ticker"].str.upper().str.contains('|'.join(search_terms), na=False)]
    
    # Min records filter
    if filters.get("min_records", 0) > 0:
        filtered_df = filtered_df[filtered_df["records"] >= filters["min_records"]]
    
    # Sort
    sort_by = filters["sort_by"]
    if sort_by == "Ticker (A-Z)":
        filtered_df = filtered_df.sort_values("ticker", ascending=True)
    elif sort_by == "Ticker (Z-A)":
        filtered_df = filtered_df.sort_values("ticker", ascending=False)
    elif sort_by == "Records (High-Low)":
        filtered_df = filtered_df.sort_values("records", ascending=False, na_position='last')
    elif sort_by == "Records (Low-High)":
        filtered_df = filtered_df.sort_values("records", ascending=True, na_position='last')
    elif sort_by == "Last Updated":
        filtered_df = filtered_df.sort_values("last_updated", ascending=False, na_position='last')
    
    # Handle bulk actions
    bulk_actions = filters.get("bulk_actions", {})
    if bulk_actions.get("select_all"):
        table_state["selected_tickers"] = filtered_df["ticker"].tolist()
        st.rerun()
    elif bulk_actions.get("deselect_all"):
        table_state["selected_tickers"] = []
        st.rerun()
    elif bulk_actions.get("select_missing"):
        table_state["selected_tickers"] = filtered_df[filtered_df["status"] == "missing"]["ticker"].tolist()
        st.rerun()
    elif bulk_actions.get("invert"):
        current = set(table_state["selected_tickers"])
        all_tickers = set(filtered_df["ticker"].tolist())
        table_state["selected_tickers"] = list(all_tickers - current)
        st.rerun()
    
    # Table container
    st.markdown('<div class="inventory-table">', unsafe_allow_html=True)
    
    # Table header
    st.markdown(f"""
    <div class="table-header">
        <div class="table-info">
            Showing <strong>{len(filtered_df)}</strong> of <strong>{len(df)}</strong> stocks
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare display dataframe
    display_df = filtered_df.copy()
    display_df["select"] = display_df["ticker"].isin(table_state["selected_tickers"])
    
    # Format columns
    if "size_kb" in display_df.columns:
        display_df["size"] = display_df["size_kb"].apply(lambda x: f"{x:.1f} KB" if pd.notna(x) else "None")
    else:
        display_df["size"] = "None"
    
    if "date_start" in display_df.columns and "date_end" in display_df.columns:
        display_df["date_range"] = display_df.apply(
            lambda row: f"{row['date_start']} to {row['date_end']}" if pd.notna(row['date_start']) else "None",
            axis=1
        )
    else:
        display_df["date_range"] = "None"
    
    # Format status with tags
    def format_status(status):
        if status == "downloaded":
            return '<span class="status-tag status-downloaded">Downloaded</span>'
        else:
            return '<span class="status-tag status-missing">Missing</span>'
    
    # Select columns to display
    table_columns = ["select", "ticker", "records", "size", "date_range", "last_updated", "status"]
    available_columns = [col for col in table_columns if col in display_df.columns or col in ["select", "size", "date_range"]]
    
    display_df = display_df[available_columns]
    display_df.columns = ["Select", "Ticker", "Records", "Size", "Date Range", "Last Updated", "Status"]
    
    # Interactive data editor
    edited_df = st.data_editor(
        display_df,
        disabled=["Ticker", "Records", "Size", "Date Range", "Last Updated", "Status"],
        hide_index=True,
        use_container_width=True,
        height=400,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=False, width="small"),
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Records": st.column_config.NumberColumn("Records", format="%d", width="small"),
            "Size": st.column_config.TextColumn("Size", width="small"),
            "Date Range": st.column_config.TextColumn("Date Range", width="medium"),
            "Last Updated": st.column_config.TextColumn("Last Updated", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="small")
        }
    )
    
    # Update selection
    table_state["selected_tickers"] = edited_df[edited_df["Select"]]["Ticker"].tolist()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update session state
    st.session_state[state_key] = table_state
    
    return {
        "selected_tickers": table_state["selected_tickers"],
        "filtered_count": len(filtered_df),
        "total_count": len(df)
    }

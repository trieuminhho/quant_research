"""Bulk actions bar component."""
import streamlit as st


def render(selected_rows: list) -> dict:
    """Render floating bulk actions bar.
    
    Args:
        selected_rows: List of selected ticker names
        
    Returns:
        Dictionary with action button states
    """
    if not selected_rows:
        return {}
    
    # Floating action bar
    st.markdown(f"""
    <div class="bulk-actions-bar">
        <span class="bulk-count">{len(selected_rows)} selected</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons in columns (below the floating bar indicator)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    
    actions = {}
    
    with col1:
        actions["download"] = st.button(
            f"📥 Download ({len(selected_rows)})",
            key="bulk_download",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        actions["validate"] = st.button(
            f"✅ Validate ({len(selected_rows)})",
            key="bulk_validate",
            use_container_width=True
        )
    
    with col3:
        actions["delete"] = st.button(
            f"🗑️ Delete ({len(selected_rows)})",
            key="bulk_delete",
            type="secondary",
            use_container_width=True
        )
    
    with col4:
        actions["clear"] = st.button(
            "❌ Clear Selection",
            key="bulk_clear",
            use_container_width=True
        )
    
    return actions

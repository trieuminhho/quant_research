"""
Common reusable components for all pages.
"""
import streamlit as st
from typing import List, Dict, Any


def render_page_header(title: str, subtitle: str = ""):
    """Render a consistent page header."""
    st.markdown(f'<div class="sub-header title-glow">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")


def render_metric_cards(metrics: List[Dict[str, Any]], columns: int = 4):
    """
    Render metric cards in a grid layout.
    
    Args:
        metrics: List of dicts with keys: label, value, delta (optional)
        columns: Number of columns in the grid
    """
    cols = st.columns(columns)
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta')
            )


def render_info_box(message: str, box_type: str = "info"):
    """
    Render an info/warning/error/success box.
    
    Args:
        message: Message to display
        box_type: One of 'info', 'warning', 'error', 'success'
    """
    box_map = {
        'info': st.info,
        'warning': st.warning,
        'error': st.error,
        'success': st.success
    }
    box_func = box_map.get(box_type, st.info)
    box_func(message)


def render_data_table(df, key: str = "table", height: int = 400):
    """Render a data table with consistent styling."""
    st.dataframe(df, use_container_width=True, height=height, key=key)


def render_loading_spinner(message: str = "Loading..."):
    """Context manager for loading spinner."""
    return st.spinner(message)


def confirm_action(message: str, button_text: str = "Confirm", key: str = None) -> bool:
    """
    Display a confirmation dialog.
    
    Returns:
        True if confirmed, False otherwise
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.warning(message)
    with col2:
        if st.button(button_text, type="primary", key=key):
            return True
    return False


def render_status_badge(status: str, label: str = None) -> str:
    """
    Render a status badge HTML.
    
    Args:
        status: One of 'success', 'warning', 'error', 'info'
        label: Optional label text
    
    Returns:
        HTML string for badge
    """
    colors = {
        'success': 'var(--success)',
        'warning': 'var(--warn)',
        'error': 'var(--danger)',
        'info': 'var(--accent-2)'
    }
    color = colors.get(status, 'var(--muted)')
    text = label or status.upper()
    
    return f'''
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        border: 1px solid {color};
        background-color: rgba(255,255,255,0.03);
        color: {color};
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    ">{text}</span>
    '''

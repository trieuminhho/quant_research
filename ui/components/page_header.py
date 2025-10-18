"""Page header component."""
import streamlit as st


def render(title: str, subtitle: str) -> tuple[bool, bool, bool]:
    """Render page header with actions.
    
    Args:
        title: Page title
        subtitle: Page subtitle/badge
        
    Returns:
        Tuple of (refresh_clicked, presets_clicked, deploy_clicked)
    """
    st.markdown(f"""
    <div class="page-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span class="page-title">{title}</span>
                <span class="page-subtitle">{subtitle}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons in columns
    col1, col2, col3, spacer = st.columns([1, 1, 1, 10])
    
    with col1:
        refresh_clicked = st.button("🔄 Refresh", key="header_refresh", use_container_width=True)
    
    with col2:
        presets_clicked = st.button("⭐ Presets", key="header_presets", use_container_width=True)
    
    with col3:
        deploy_clicked = st.button("🚀 Deploy", key="header_deploy", use_container_width=True)
    
    return refresh_clicked, presets_clicked, deploy_clicked

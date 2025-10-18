"""Sidebar navigation component."""
import streamlit as st


def render(active: str) -> None:
    """Render sidebar navigation.
    
    Args:
        active: Currently active page name
    """
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="padding: 16px 0; border-bottom: 1px solid var(--border); margin-bottom: 16px;">
            <div style="font-size: 20px; font-weight: 700; color: var(--text);">
                📊 QuantSearch
            </div>
            <div style="font-size: 12px; color: var(--muted); margin-top: 4px; letter-spacing: 0.06em;">
                Trading Platform
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation items
        pages = [
            ("Home", "🏠"),
            ("Data Management", "💾"),
            ("Stock Data Viewer", "📈"),
            ("Feature Engineering", "🔧"),
            ("Model Training", "🤖"),
            ("Strategy Backtesting", "🎯"),
            ("Performance Analysis", "📊"),
            ("System Configuration", "⚙️")
        ]
        
        for page_name, icon in pages:
            is_active = page_name == active
            
            # Active state styling
            if is_active:
                st.markdown(f"""
                <div style="
                    background: rgba(139,92,246,0.18);
                    border-left: 3px solid var(--accent);
                    padding: 12px 16px;
                    margin: 4px -16px 4px 0;
                    cursor: pointer;
                    border-radius: 0 8px 8px 0;
                ">
                    <span style="font-size: 16px; margin-right: 8px;">{icon}</span>
                    <span style="font-weight: 600; color: var(--accent); letter-spacing: 0.04em;">{page_name}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    padding: 12px 16px;
                    margin: 4px -16px 4px 0;
                    cursor: pointer;
                    border-radius: 0 8px 8px 0;
                    transition: background 0.2s;
                " onmouseover="this.style.background='rgba(139,92,246,0.12)'" 
                  onmouseout="this.style.background='transparent'">
                    <span style="font-size: 16px; margin-right: 8px; opacity: 0.6;">{icon}</span>
                    <span style="color: var(--muted); letter-spacing: 0.02em;">{page_name}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style="
            position: fixed;
            bottom: 16px;
            font-size: 11px;
            color: var(--muted);
            padding: 0 16px;
            letter-spacing: 0.08em;
        ">
            v2.0.0 • © 2025
        </div>
        """, unsafe_allow_html=True)

"""KPI cards component."""
import streamlit as st


def render(stats: dict, state_key: str = "kpi_filter") -> None:
    """Render KPI cards that can filter the table.
    
    Args:
        stats: Dictionary with keys: downloaded, not_downloaded, total_records, universe
        state_key: Session state key for active filter
    """
    # Initialize session state
    if state_key not in st.session_state:
        st.session_state[state_key] = None
    
    col1, col2, col3, col4 = st.columns(4)
    
    cards = [
        {
            "col": col1,
            "icon": "📥",
            "label": "Downloaded",
            "value": stats.get("downloaded", 0),
            "annotation": f"{stats.get('downloaded_pct', 0):.1f}%",
            "filter": "downloaded"
        },
        {
            "col": col2,
            "icon": "❌",
            "label": "Not Downloaded",
            "value": stats.get("not_downloaded", 0),
            "annotation": f"{stats.get('not_downloaded_pct', 0):.1f}%",
            "filter": "not_downloaded"
        },
        {
            "col": col3,
            "icon": "📊",
            "label": "Total Records",
            "value": f"{stats.get('total_records', 0):,}",
            "annotation": "OHLCV data points",
            "filter": None
        },
        {
            "col": col4,
            "icon": "🎯",
            "label": "Universe",
            "value": stats.get("universe", 0),
            "annotation": "S&P 500",
            "filter": None
        }
    ]
    
    for card in cards:
        with card["col"]:
            # Check if this card is active
            is_active = st.session_state[state_key] == card["filter"]
            active_class = "active" if is_active else ""
            
            # Create clickable card
            card_html = f"""
            <div class="kpi-card {active_class}" onclick="document.getElementById('kpi_{card['filter']}').click()">
                <div class="kpi-label">
                    <span class="kpi-icon">{card['icon']}</span>
                    {card['label']}
                </div>
                <div class="kpi-value">{card['value']}</div>
                <div class="kpi-annotation">{card['annotation']}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Hidden button for state management
            if card["filter"] and st.button("", key=f"kpi_{card['filter']}", type="secondary"):
                # Toggle filter
                if st.session_state[state_key] == card["filter"]:
                    st.session_state[state_key] = None
                else:
                    st.session_state[state_key] = card["filter"]
                st.rerun()

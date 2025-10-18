"""Dark theme styling helpers for the QuantSearch Streamlit app."""

import streamlit as st

CSS = """
:root {
  --bg: #0B0F1A;
  --bg-2: #0F172A;
  --card: #0D1320;
  --muted: #94A3B8;
  --text: #E5E7EB;
  --border: #1F2937;
  --accent: #8B5CF6;
  --accent-2: #22D3EE;
  --success: #10B981;
  --warn: #F59E0B;
  --danger: #EF4444;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
}

[data-testid="stHeader"] {
  background: linear-gradient(180deg, rgba(13,19,32,0.75), rgba(13,19,32,0.35));
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--border);
}

section.main > div {
  padding-top: 0.5rem;
}

h1, h2, h3, h4 {
  color: var(--text);
  letter-spacing: 0.2px;
}

.stMetric, .stMetric label, .stMetricValue {
  color: var(--text) !important;
}

.block-container {
  padding-top: 1.25rem;
  padding-bottom: 2rem;
  max-width: 1320px;
}

div[role="radiogroup"] label,
.stSelectbox, .stMultiSelect,
.stTextInput, .stDateInput {
  color: var(--text);
}

.stButton>button, .stDownloadButton>button {
  border: 1px solid var(--border);
  background: radial-gradient(1200px circle at 0 0, rgba(139,92,246,0.15), transparent 40%),
              var(--bg-2);
  color: var(--text);
  box-shadow: 0 0 0 0 rgba(139,92,246,0.35);
  transition: box-shadow 0.2s ease, transform 0.05s ease, border-color 0.2s ease;
}

.stButton>button:hover, .stDownloadButton>button:hover {
  border-color: var(--accent);
  box-shadow: 0 8px 30px -10px rgba(139,92,246,0.45);
}

.stButton>button:active {
  transform: translateY(1px);
}

.st-cq, .stDataFrame, .stTable, [data-testid="stDataFrameResizable"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}

.stDataFrame [role="row"]:nth-child(odd) {
  background: rgba(255,255,255,0.02);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0C1220, #0A0F1A);
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] div[role="radiogroup"] {
  gap: 0.5rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
  background: transparent;
  padding: 0.75rem 1rem;
  border-radius: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid transparent;
  font-weight: 500;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(139,92,246,0.12);
  border-color: var(--accent);
  color: var(--text) !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
  background: rgba(139,92,246,0.18);
  border-color: var(--accent);
  color: var(--text) !important;
  box-shadow: inset 0 0 0 1px rgba(139,92,246,0.35);
}

[data-testid="stSidebarNav"] a, [data-testid="stSidebar"] * {
  color: var(--muted) !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
  color: var(--text) !important;
}

.tag, .pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.15rem 0.55rem;
  border: 1px solid var(--border);
  border-radius: 9999px;
  background: rgba(255,255,255,0.03);
  color: var(--muted);
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px -18px rgba(0,0,0,0.7);
}

.kpi {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(180deg, rgba(34,211,238,0.08), rgba(34,211,238,0));
}

.kpi h3 {
  margin: 0;
  font-weight: 600;
  color: var(--muted);
  font-size: 0.85rem;
}

.kpi .value {
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--text);
}

.kpi .delta.up {
  color: var(--success);
}

.kpi .delta.down {
  color: var(--danger);
}

hr, .stDivider {
  border-color: var(--border) !important;
}

.sub-header {
  font-size: 1.45rem;
  font-weight: 600;
  color: var(--text);
  margin: 1.25rem 0 0.65rem;
}

.filter-panel {
  background: rgba(13,19,32,0.85);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.35rem 1.5rem 1.1rem;
  box-shadow: 0 18px 44px -32px rgba(0,0,0,0.85);
  backdrop-filter: blur(10px);
}

.filter-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.8rem;
  border-radius: 9999px;
  border: 1px solid rgba(139,92,246,0.35);
  background: rgba(139,92,246,0.16);
  color: var(--text);
  font-size: 0.8rem;
  letter-spacing: 0.02em;
}

.filter-chip-close {
  cursor: pointer;
  opacity: 0.65;
  transition: opacity 0.2s ease;
}

.filter-chip-close:hover {
  opacity: 1;
}

.metric-card {
  background: linear-gradient(180deg, rgba(139,92,246,0.12), rgba(11,15,26,0.85));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem 1.25rem;
  box-shadow: 0 12px 22px -18px rgba(0,0,0,0.75);
}

.kpi-card {
  position: relative;
  background: linear-gradient(160deg, rgba(139,92,246,0.16), rgba(34,211,238,0.08));
  border: 1px solid rgba(139,92,246,0.35);
  border-radius: 18px;
  padding: 1.1rem 1.25rem;
  margin-bottom: 0.8rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.3s ease, border-color 0.3s ease;
  box-shadow: 0 22px 40px -32px rgba(139,92,246,0.55);
}

.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 35px 60px -34px rgba(139,92,246,0.75);
}

.kpi-card.active {
  border-color: var(--accent);
  box-shadow: 0 38px 70px -32px rgba(139,92,246,0.9);
}

.kpi-label {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--muted);
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.kpi-icon {
  font-size: 0.95rem;
}

.kpi-value {
  font-size: 2.1rem;
  font-weight: 700;
  color: var(--text);
  margin-top: 0.35rem;
}

.kpi-annotation {
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: 0.35rem;
}

.validation-success,
.validation-warning,
.validation-error {
  padding: 0.85rem 1rem;
  border-left: 3px solid transparent;
  border-radius: 12px;
  background: rgba(255,255,255,0.02);
  margin-bottom: 0.65rem;
  color: var(--text);
}

.validation-success {
  border-color: var(--success);
  background: linear-gradient(120deg, rgba(16,185,129,0.15), rgba(11,15,26,0.6));
}

.validation-warning {
  border-color: var(--warn);
  background: linear-gradient(120deg, rgba(245,158,11,0.15), rgba(11,15,26,0.6));
}

.validation-error {
  border-color: var(--danger);
  background: linear-gradient(120deg, rgba(239,68,68,0.16), rgba(11,15,26,0.6));
}

.source-match {
  color: var(--success);
  font-weight: 600;
}

.source-mismatch {
  color: var(--danger);
  font-weight: 600;
}

.source-unavailable {
  color: var(--muted);
  font-style: italic;
}

.inventory-table {
  background: rgba(13,19,32,0.88);
  border: 1px solid var(--border);
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 25px 58px -45px rgba(0,0,0,0.8);
}

.table-header {
  padding: 1rem 1.4rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(15,23,42,0.78);
  backdrop-filter: blur(8px);
}

.table-info {
  color: var(--muted);
  font-size: 0.9rem;
  letter-spacing: 0.03em;
}

.status-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.65rem;
  border-radius: 9999px;
  font-weight: 600;
  font-size: 0.75rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.status-downloaded {
  background: rgba(16,185,129,0.2);
  color: var(--success);
  border: 1px solid rgba(16,185,129,0.45);
}

.status-missing {
  background: rgba(239,68,68,0.18);
  color: var(--danger);
  border: 1px solid rgba(239,68,68,0.4);
}

.bulk-actions-bar {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(11,15,26,0.94);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.85rem 1.4rem;
  box-shadow: 0 30px 60px -34px rgba(0,0,0,0.9);
  display: flex;
  align-items: center;
  gap: 1rem;
  z-index: 1000;
  backdrop-filter: blur(12px);
}

.bulk-count {
  font-weight: 600;
  color: var(--text);
  padding: 0.25rem 0.7rem;
  border-radius: 9999px;
  border: 1px solid rgba(139,92,246,0.35);
  background: rgba(139,92,246,0.15);
  letter-spacing: 0.08em;
}

.page-header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: linear-gradient(180deg, rgba(13,19,32,0.92), rgba(13,19,32,0.55));
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--border);
  margin: -1.25rem -0.5rem 1.5rem;
  border-radius: 0 0 18px 18px;
  box-shadow: 0 26px 60px -42px rgba(0,0,0,0.85);
  backdrop-filter: blur(16px);
}

.page-title {
  font-size: 1.6rem;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.01em;
}

.page-subtitle {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  margin-left: 0.75rem;
  padding: 0.2rem 0.7rem;
  border-radius: 9999px;
  border: 1px solid rgba(34,211,238,0.45);
  background: rgba(34,211,238,0.12);
  color: var(--accent-2);
  font-size: 0.85rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
"""


def apply_dark_theme() -> None:
    """Inject the dark theme CSS once per page."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .title-glow {
          text-shadow: 0 0 18px rgba(139,92,246,0.45), 0 0 2px rgba(34,211,238,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

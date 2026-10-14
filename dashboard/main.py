"""Named Streamlit navigation for the store and cannibalization dashboards."""

from pathlib import Path

import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CU Store Analytics",
    page_icon="🏪",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            DASHBOARD_DIR / "app.py",
            title="Store Performance",
            default=True,
        ),
        st.Page(
            DASHBOARD_DIR / "pages" / "2_Cannibalization.py",
            title="Cannibalization Analysis",
        ),
    ],
    position="sidebar",
)
page.run()

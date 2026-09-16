"""Shared UI tweaks for the TokenWise dashboard."""
import streamlit as st

_BLUE = "#1f77b4"


def apply_theme() -> None:
    """Recolor Streamlit's top decoration bar to match the blue primary color."""
    st.markdown(
        f"""
        <style>
        [data-testid="stDecoration"] {{
            background: {_BLUE} !important;
            background-image: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

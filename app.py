"""PixelOpt -- Streamlit entry point.

This file is only the frame: page config, the shared styling, and top
navigation. Each tool lives in its own page under app_pages/, and anything
the pages share lives in app_shared.py and ui_components.py.

    streamlit run app.py
"""

import streamlit as st

from ui_components import app_style

st.set_page_config(
    page_title="PixelOpt",
    page_icon=":material/compress:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page("app_pages/optimize.py", title="Optimize",
                icon=":material/auto_awesome:", default=True),
        st.Page("app_pages/form_photo.py", title="Form photo", icon=":material/badge:"),
        st.Page("app_pages/batch.py", title="Batch", icon=":material/photo_library:"),
    ],
    position="top",
)

app_style()
page.run()

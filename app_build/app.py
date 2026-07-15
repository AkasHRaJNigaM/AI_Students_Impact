import streamlit as st

from lib.style import inject_css

st.set_page_config(
    page_title="AI Impact on Students",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

PAGES = {
    "Overview": "pages_content.overview",
    "Explore the Data": "pages_content.eda",
    "Feature Engineering": "pages_content.feature_eng",
    "Predict: Academic Performance": "pages_content.predict_regression",
    "Predict: Burnout Risk": "pages_content.predict_burnout",
    "Model Cards": "pages_content.model_cards",
}

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 6px 4px 18px 4px;">
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.15rem; color:#F5F3FF;">
                🎓 AI × Student Impact
            </div>
            <div style="color:#7A8096; font-size:0.8rem; margin-top:2px;">
                GenAI usage vs. academic outcomes
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#5C6178; font-size:0.75rem; line-height:1.5;">
        Models are pre-trained offline and loaded read-only.<br>
        No training happens in this app.
        </div>
        """,
        unsafe_allow_html=True,
    )

import importlib

module = importlib.import_module(PAGES[choice])
module.render()

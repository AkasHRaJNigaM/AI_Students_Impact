import plotly.express as px
import streamlit as st

from lib.inference import predict_burnout_level, predict_is_burnout
from lib.models import load_burnout_level_assets, load_is_burnout_assets
from lib.style import ACCENT_SEQUENCE, PLOTLY_TEMPLATE, hero, metric_card, risk_pill, section_tag


def _shared_inputs(key_prefix: str, include_gpa: bool):
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.selectbox("Year of Study", ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
                             index=3, key=f"{key_prefix}_year")
        anxiety = st.slider("Anxiety Level During Exams (1-10)", 1, 10, 5, key=f"{key_prefix}_anx")
    with c2:
        weekly_ai = st.slider("Weekly GenAI Hours", 0.0, 40.0, 10.0, 0.5, key=f"{key_prefix}_wai")
        tool_div = st.slider("Tool Diversity (# distinct tools)", 1, 5, 3, key=f"{key_prefix}_tool")
    with c3:
        skill = st.selectbox("Prompt Engineering Skill", ["Beginner", "Intermediate", "Advanced"],
                              index=1, key=f"{key_prefix}_skill")
        trad_hours = st.slider("Traditional Study Hours / week", 0.0, 32.0, 10.0, 0.5, key=f"{key_prefix}_trad")

    c4, c5, c6 = st.columns(3)
    with c4:
        dependency = st.slider("Perceived AI Dependency (1-10)", 1, 10, 5, key=f"{key_prefix}_dep")
    with c5:
        retention = st.slider("Skill Retention Score (0-100)", 0, 100, 75, key=f"{key_prefix}_ret")
    pre_gpa = None
    if include_gpa:
        with c6:
            pre_gpa = st.slider("Pre-Semester GPA", 1.0, 4.0, 3.2, 0.01, key=f"{key_prefix}_gpa")

    inputs = {
        "Year_of_Study": year,
        "Anxiety_Level_During_Exams": anxiety,
        "Weekly_GenAI_Hours": weekly_ai,
        "Tool_Diversity": tool_div,
        "Prompt_Engineering_Skill": skill,
        "Traditional_Study_Hours": trad_hours,
        "Perceived_AI_Dependency": dependency,
        "Skill_Retention_Score": retention,
    }
    if include_gpa:
        inputs["Pre_Semester_GPA"] = pre_gpa
    return inputs


def _render_level_tab():
    assets = load_burnout_level_assets()
    method = st.radio(
        "Model", ["Multinomial Logistic Regression", "Decision Tree"],
        horizontal=True, key="level_method",
    )
    with st.form("burnout_level_form"):
        section_tag("Student profile")
        inputs = _shared_inputs("lvl", include_gpa=False)
        submitted = st.form_submit_button("Predict Burnout Risk Level")

    if not submitted:
        st.info("Fill in the profile above and click **Predict Burnout Risk Level**.")
        return

    result = predict_burnout_level(inputs, assets, method)
    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Prediction")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### {risk_pill(result['pred_class'])}", unsafe_allow_html=True)
        st.caption(f"Predicted with: {method}")
    with c2:
        probs = result["probabilities"]
        order = ["Low", "Medium", "High"]
        fig = px.bar(
            x=[probs[k] for k in order], y=order, orientation="h",
            template=PLOTLY_TEMPLATE, color=order,
            color_discrete_map={"Low": "#6EE7B7", "Medium": "#FACC15", "High": "#FCA5A5"},
            labels={"x": "Probability", "y": ""},
        )
        fig.update_layout(height=240, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, width='stretch')


def _render_binary_tab():
    assets = load_is_burnout_assets()
    method = st.radio(
        "Model", ["Logistic Regression", "Tuned Decision Tree"],
        horizontal=True, key="binary_method",
    )
    with st.form("is_burnout_form"):
        section_tag("Student profile")
        inputs = _shared_inputs("bin", include_gpa=True)
        submitted = st.form_submit_button("Predict High-Burnout Flag")

    if not submitted:
        st.info("Fill in the profile above and click **Predict High-Burnout Flag**.")
        return

    result = predict_is_burnout(inputs, assets, method)
    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        label = "High Burnout Risk" if result["flag"] else "Not High Burnout"
        pill_level = "High" if result["flag"] else "Low"
        st.markdown(f"### {risk_pill(pill_level)}", unsafe_allow_html=True)
        st.caption(label)
    with c2:
        metric_card("Predicted Probability", f"{result['probability']*100:.1f}%", "of severe burnout")
    with c3:
        metric_card("Decision Threshold", f"{result['threshold']*100:.1f}%",
                     "tuned via Youden's J statistic, not a default 50%")

    st.progress(min(max(result["probability"], 0.0), 1.0))
    if method == "Tuned Decision Tree":
        st.caption(f"Cross-validated ROC-AUC during tuning: {assets.tree_cv_auc:.3f}")


def render():
    hero(
        "Burnout Risk Classification",
        "Estimate a student's burnout risk",
        "Two complementary views: a 3-level risk classifier (Low/Medium/High) and a "
        "binary high-burnout flag tuned for recall using a custom probability threshold.",
    )
    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["3-level Risk (Low/Medium/High)", "Binary High-Burnout Flag"])
    with tab1:
        _render_level_tab()
    with tab2:
        _render_binary_tab()

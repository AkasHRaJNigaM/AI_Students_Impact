import plotly.graph_objects as go
import streamlit as st

from lib.inference import predict_gpa
from lib.models import load_regression_assets
from lib.style import PLOTLY_TEMPLATE, hero, metric_card, section_tag


def render():
    hero(
        "Linear Regression",
        "Predict a student's post-semester GPA",
        "An OLS model (fit with HC3 robust standard errors) trained on study habits, "
        "GenAI usage, and prior academic performance.",
    )
    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    assets = load_regression_assets()

    with st.form("regression_form"):
        section_tag("Student profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            year = st.selectbox("Year of Study", ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"], index=3)
            pre_gpa = st.slider("Pre-Semester GPA", 1.0, 4.0, 3.2, 0.01)
        with c2:
            weekly_ai = st.slider("Weekly GenAI Hours", 0.0, 40.0, 8.0, 0.5)
            skill = st.selectbox("Prompt Engineering Skill", ["Beginner", "Intermediate", "Advanced"], index=1)
        with c3:
            trad_hours = st.slider("Traditional Study Hours / week", 0.0, 32.0, 11.0, 0.5)
            use_case = st.selectbox(
                "Primary AI Use Case",
                ["Copywriting/Drafting", "Ideation", "Summarizing_Reading",
                 "Debugging/Troubleshooting", "Direct_Answer_Generation"],
                index=3,
            )

        c4, c5 = st.columns(2)
        with c4:
            dependency = st.slider("Perceived AI Dependency (1-10)", 1, 10, 4)
        with c5:
            anxiety = st.slider("Anxiety Level During Exams (1-10)", 1, 10, 4)

        submitted = st.form_submit_button("Predict GPA")

    if not submitted:
        st.info("Fill in the profile above and click **Predict GPA**.")
        return

    raw_inputs = {
        "Year_of_Study": year,
        "Pre_Semester_GPA": pre_gpa,
        "Weekly_GenAI_Hours": weekly_ai,
        "Prompt_Engineering_Skill": skill,
        "Traditional_Study_Hours": trad_hours,
        "Perceived_AI_Dependency": dependency,
        "Anxiety_Level_During_Exams": anxiety,
        "Primary_Use_Case": use_case,
    }
    result = predict_gpa(raw_inputs, assets)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Predicted Post-Semester GPA", f"{result['prediction']:.2f}", "on a 0-4 scale")
    with c2:
        change = result["gpa_change"]
        metric_card(
            "Change vs. Pre-Semester GPA",
            f"{change:+.2f}",
            "predicted trajectory",
            delta=("↑ improving" if change >= 0 else "↓ declining"),
            positive=change >= 0,
        )
    with c3:
        metric_card(
            "96% Prediction Interval",
            f"{result['ci_low']:.2f} – {result['ci_high']:.2f}",
            "plausible range for this individual student",
        )

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=result["prediction"],
        number={"suffix": " GPA"},
        gauge={
            "axis": {"range": [0, 4]},
            "bar": {"color": "#7C6FF2"},
            "steps": [
                {"range": [0, 2], "color": "rgba(252,165,165,0.25)"},
                {"range": [2, 3], "color": "rgba(250,204,21,0.2)"},
                {"range": [3, 4], "color": "rgba(110,231,183,0.25)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": pre_gpa},
        },
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=300, margin=dict(t=30, b=10, l=30, r=30))
    st.plotly_chart(fig, width='stretch')
    st.caption("White marker shows the entered Pre-Semester GPA for reference.")

    with st.expander("Model coefficients (standardized)"):
        params = assets.model.params.drop("const")
        st.bar_chart(params)
        st.caption(
            "Coefficients are on standardized (z-scored) features, so magnitude reflects "
            "relative influence — Pre-Semester GPA and Traditional Study Hours dominate."
        )

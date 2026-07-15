import streamlit as st

from lib.data import load_raw_data
from lib.style import hero, metric_card, section_tag


def render():
    df = load_raw_data()

    hero(
        "AI Student Impact Dataset · 50,000 students",
        "How GenAI habits shape academic outcomes",
        "An interactive dashboard for exploring the dataset, understanding the feature "
        "engineering behind the models, and running live predictions with the trained "
        "regression and classification models — no retraining, ever.",
    )

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Students", f"{len(df):,}", "rows in the raw dataset")
    with c2:
        metric_card("Avg. Pre-Semester GPA", f"{df['Pre_Semester_GPA'].mean():.2f}", "on a 0-4 scale")
    with c3:
        metric_card("Avg. Weekly GenAI Hours", f"{df['Weekly_GenAI_Hours'].mean():.1f} hrs", "self-reported usage")
    with c4:
        high_share = (df["Burnout_Risk_Level"] == "High").mean() * 100
        metric_card("High Burnout Risk", f"{high_share:.1f}%", "of students", )

    st.write("")
    section_tag("What's inside")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **📊 Explore the Data** — distributions, outliers, and categorical
            breakdowns straight from the raw survey data.

            **🧪 Feature Engineering** — the exact encoding, scaling, and
            engineered-feature steps used before any model saw the data,
            plus correlation and multicollinearity (VIF) checks.
            """
        )
    with col2:
        st.markdown(
            """
            **📈 Predict: Academic Performance** — a linear regression (OLS)
            model estimating a student's post-semester GPA from their study
            and AI-usage habits.

            **⚠️ Predict: Burnout Risk** — logistic models (multinomial and
            binary) estimating burnout risk level from engagement and
            workload signals.
            """
        )

    st.write("")
    section_tag("Sample of the raw data")
    st.dataframe(df.sample(8, random_state=7), width='stretch', hide_index=True)

import pandas as pd
import plotly.express as px
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import statsmodels.api as sm

from lib.data import (
    CLASSIFICATION_PREDICTORS,
    REGRESSION_DROP_FOR_TRAIN,
    REG_DROPPED_DUMMIES,
    REGRESSION_TARGET,
    build_classification_frame,
    build_encoded_frame,
    load_raw_data,
)
from lib.style import PLOTLY_TEMPLATE, hero, section_tag


@st.cache_data(show_spinner=False)
def _vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(frame)
    out = pd.DataFrame({"Feature": X.columns})
    out["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
    return out.sort_values("VIF").reset_index(drop=True)


def render():
    hero(
        "Feature Engineering",
        "From raw survey answers to model-ready features",
        "Every step below mirrors the notebook exactly, so predictions made in this app "
        "are consistent with how the models were trained and evaluated.",
    )
    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    section_tag("Encoding steps")
    st.markdown(
        """
        - **Ordinal mapping** — `Year_of_Study` (Freshman→1 ... Graduate→5), `Prompt_Engineering_Skill`
          (Beginner→1, Intermediate→2, Advanced→3), and `Burnout_Risk_Level` (Low→1, Medium→2, High→3)
          are mapped to ordered integers rather than one-hot encoded, preserving their natural ordering.
        - **One-hot encoding** — `Major_Category`, `Primary_Use_Case`, and `Institutional_Policy` are
          nominal (unordered), so they're expanded into dummy columns. One reference category per
          group is dropped afterward to control multicollinearity:
          `Major_Category_Arts`, `Primary_Use_Case_Ideation`, `Institutional_Policy_Actively_Encouraged`.
        - **Outlier removal** — rows with `Traditional_Study_Hours` > 32/week are dropped before any
          modelling.
        - **Engineered interaction features** (used by the burnout models only):
          `AI_Engagement = Weekly_GenAI_Hours × Tool_Diversity × Prompt_Engineering_Skill`,
          `AI_Dependency_Ratio = Weekly_GenAI_Hours / (Traditional_Study_Hours + 1)`,
          `Total_Study = Weekly_GenAI_Hours + Traditional_Study_Hours`.
        - **Engineered interaction feature** (regression only): `Depending_Hours`, the mean-centered
          product of `Weekly_GenAI_Hours` and `Perceived_AI_Dependency`.
        - **Standardization** — every numeric feature is scaled with `StandardScaler` before it reaches
          a linear/logistic model.
        """
    )

    df = load_raw_data()
    encoded = build_encoded_frame(df)
    classification_frame = build_classification_frame(df)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Correlation with the targets")
    tab1, tab2 = st.tabs(["vs. Post_Semester_GPA", "vs. Burnout_Risk_Level"])
    with tab1:
        corr = encoded.corr(numeric_only=True)[REGRESSION_TARGET].drop(REGRESSION_TARGET).sort_values()
        fig = px.bar(
            corr, orientation="h", template=PLOTLY_TEMPLATE,
            labels={"value": "Pearson r", "index": ""}, color=corr.values,
            color_continuous_scale="RdBu", color_continuous_midpoint=0,
        )
        fig.update_layout(height=520, showlegend=False, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    with tab2:
        corr2 = encoded.corr(numeric_only=True)["Burnout_Risk_Level"].drop("Burnout_Risk_Level").sort_values()
        fig2 = px.bar(
            corr2, orientation="h", template=PLOTLY_TEMPLATE,
            labels={"value": "Pearson r", "index": ""}, color=corr2.values,
            color_continuous_scale="RdBu", color_continuous_midpoint=0,
        )
        fig2.update_layout(height=520, showlegend=False, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, width='stretch')

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Multicollinearity check (VIF)")
    st.caption("Variance Inflation Factor on each model's final training features. Values above ~5 flag concerning collinearity.")

    vt1, vt2 = st.tabs(["Regression features", "Burnout classifier features"])
    with vt1:
        reg_frame = encoded.drop(columns=REG_DROPPED_DUMMIES).drop(columns=REGRESSION_DROP_FOR_TRAIN, errors="ignore")
        st.dataframe(_vif_table(reg_frame), width='stretch', hide_index=True)
    with vt2:
        st.dataframe(_vif_table(classification_frame[CLASSIFICATION_PREDICTORS]), width='stretch', hide_index=True)

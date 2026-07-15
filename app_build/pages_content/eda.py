import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.data import build_encoded_frame, load_raw_data
from lib.style import ACCENT_SEQUENCE, PLOTLY_TEMPLATE, hero, section_tag


def render():
    df = load_raw_data()
    hero(
        "Exploratory Data Analysis",
        "Distributions, outliers & category breakdowns",
        "The same checks performed in the notebook's EDA section, made interactive.",
    )
    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    num_cols = df.select_dtypes(exclude=["object", "bool"]).columns.tolist()
    num_cols = [c for c in num_cols if c != "Student_ID"]
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    section_tag("Numeric distributions")
    col_select, col_chart = st.columns([1, 3])
    with col_select:
        num_col = st.selectbox("Column", num_cols, index=num_cols.index("Weekly_GenAI_Hours"))
        show_box = st.checkbox("Show boxplot (outliers)", value=True)
    with col_chart:
        fig = px.histogram(
            df, x=num_col, nbins=40, marginal="box" if show_box else None,
            template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT_SEQUENCE[0]],
        )
        fig.update_layout(height=380, margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, width='stretch')

    if num_col == "Traditional_Study_Hours":
        n_outliers = (df["Traditional_Study_Hours"] > 32).sum()
        st.info(
            f"⚠️ {n_outliers} students report more than 32 hours/week of traditional study — "
            "these rows are treated as outliers and excluded before model training."
        )

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Categorical breakdown")
    cat_col = st.selectbox("Column", cat_cols + bool_cols)
    vc = df[cat_col].value_counts().reset_index()
    vc.columns = [cat_col, "count"]
    fig2 = px.bar(
        vc, x=cat_col, y="count", template=PLOTLY_TEMPLATE,
        color=cat_col, color_discrete_sequence=ACCENT_SEQUENCE,
    )
    fig2.update_layout(height=380, showlegend=False, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig2, width='stretch')

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Correlation heatmap")
    st.caption("Pearson correlation across all encoded numeric features (after ordinal mapping & one-hot encoding).")
    encoded = build_encoded_frame(df)
    corr = encoded.corr(numeric_only=True).round(2)
    fig3 = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0, colorbar=dict(title="r"),
        )
    )
    fig3.update_layout(
        template=PLOTLY_TEMPLATE, height=650,
        margin=dict(t=20, b=10, l=10, r=10),
    )
    st.plotly_chart(fig3, width='stretch')

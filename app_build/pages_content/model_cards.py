import streamlit as st

from lib.evaluation import burnout_level_metrics, is_burnout_metrics, regression_metrics
from lib.style import hero, metric_card, section_tag


def render():
    hero(
        "Model Cards",
        "What's under the hood, and how well it performs",
        "Metrics below are computed on a held-out test split using the exact same "
        "random seed as the training notebook — models are evaluated, never retrained.",
    )
    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    section_tag("Post_Semester_GPA · Linear Regression (OLS, HC3 robust SE)")
    m = regression_metrics()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("R²", f"{m['R2']:.1f}%", "variance explained")
    with c2:
        metric_card("MAE", f"{m['MAE']:.3f}", "mean absolute error (GPA points)")
    with c3:
        metric_card("RMSE", f"{m['RMSE']:.3f}", "root mean squared error")
    with c4:
        metric_card("Test set size", f"{m['n_test']:,}", "held-out students")
    st.caption(
        "This model has a known limitation, flagged in the notebook: residuals show "
        "heteroscedasticity, likely because predictors don't fully capture variance in "
        "how post-semester GPA changes. Interpret point predictions with the confidence "
        "interval shown on the prediction page, not as an exact figure."
    )

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Burnout_Risk_Level (Low/Medium/High) · Multinomial Logit vs. Decision Tree")
    m2 = burnout_level_metrics()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Multinomial Logit Accuracy", f"{m2['log_acc']:.1f}%")
    with c2:
        metric_card("Decision Tree Accuracy", f"{m2['tree_acc']:.1f}%")
    with c3:
        metric_card("Test set size", f"{m2['n_test']:,}", "held-out students")
    st.caption(
        "3-class burnout prediction is the hardest task in this project — the Medium "
        "class overlaps heavily with both Low and High in feature space, which caps "
        "achievable accuracy. The binary framing below performs noticeably better."
    )

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_tag("Is_Burn_Out (High vs. Not High) · Logistic Regression vs. Tuned Decision Tree")
    m3 = is_burnout_metrics()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Logistic Regression ROC-AUC", f"{m3['log_auc']:.3f}")
    with c2:
        metric_card("Tuned Decision Tree ROC-AUC", f"{m3['tree_auc']:.3f}")
    with c3:
        metric_card("Test set size", f"{m3['n_test']:,}", "held-out students")
    st.caption(
        "Both binary models use a custom decision threshold (found via Youden's J "
        "statistic) instead of the default 0.5, trading a bit of precision for much "
        "better recall on the minority high-burnout class."
    )

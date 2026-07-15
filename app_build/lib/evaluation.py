"""
Recomputes held-out test metrics for each already-trained model, using the
exact same train/test split recipe (same random_state, test_size, stratify)
as the notebook. This never calls .fit() on any predictive model — only
train_test_split (a deterministic shuffle) and .predict()/.predict_proba().

Everything here is wrapped in st.cache_data / st.cache_resource so it runs
exactly once per app process, not on every page view.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from lib.data import (
    CLASSIFICATION_PREDICTORS,
    REGRESSION_DROP_FOR_TRAIN,
    REG_DROPPED_DUMMIES,
    REGRESSION_TARGET,
    build_classification_frame,
    build_encoded_frame,
    load_raw_data,
)
from lib.models import (
    load_burnout_level_assets,
    load_is_burnout_assets,
    load_regression_assets,
)


@st.cache_data(show_spinner="Scoring regression model on held-out data...")
def regression_metrics() -> dict:
    df = load_raw_data()
    encoded = build_encoded_frame(df)
    copy1 = encoded.drop(columns=REG_DROPPED_DUMMIES)
    target = copy1[REGRESSION_TARGET].copy()

    train_data = copy1.drop(columns=REGRESSION_DROP_FOR_TRAIN, errors="ignore")
    assets = load_regression_assets()
    scaler = assets.scaler
    feat_names = list(scaler.feature_names_in_)

    mean_wgh = scaler.mean_[feat_names.index("Weekly_GenAI_Hours")]
    mean_pad = scaler.mean_[feat_names.index("Perceived_AI_Dependency")]
    train_data = train_data.copy()
    train_data["Depending_Hours"] = (train_data["Weekly_GenAI_Hours"] - mean_wgh) * (
        train_data["Perceived_AI_Dependency"] - mean_pad
    )
    train_data = train_data.reindex(columns=feat_names)

    scaled = pd.DataFrame(scaler.transform(train_data), columns=feat_names, index=train_data.index)
    _, x_test, _, y_test = train_test_split(scaled, target, test_size=0.4, shuffle=True, random_state=42)

    exog_names = assets.model.model.exog_names
    x_test_sm = sm.add_constant(x_test, has_constant="add").reindex(columns=exog_names)
    preds = assets.model.predict(x_test_sm)

    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": root_mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds) * 100,
        "n_test": len(y_test),
    }


@st.cache_data(show_spinner="Scoring burnout-level classifiers on held-out data...")
def burnout_level_metrics() -> dict:
    df = load_raw_data()
    copy2 = build_classification_frame(df)
    train_data = copy2[CLASSIFICATION_PREDICTORS]
    target = copy2["Burnout_Risk_Level"]

    assets = load_burnout_level_assets()
    scaler = assets.scaler
    feat_names = list(scaler.feature_names_in_)
    scaled = pd.DataFrame(scaler.transform(train_data[feat_names]), columns=feat_names, index=train_data.index)

    _, x_test, _, y_test = train_test_split(scaled, target, stratify=target, test_size=0.4, random_state=26)

    exog_names = assets.log_model.model.exog_names
    x_test_sm = sm.add_constant(x_test, has_constant="add").reindex(columns=exog_names)
    probs = assets.log_model.predict(x_test_sm)
    log_pred = np.argmax(probs.values, axis=1) + 1
    log_acc = accuracy_score(y_test, log_pred)

    tree_pred = assets.tree_model.predict(x_test.reindex(columns=assets.tree_model.feature_names_in_))
    tree_acc = accuracy_score(y_test, tree_pred)

    return {"log_acc": log_acc * 100, "tree_acc": tree_acc * 100, "n_test": len(y_test)}


@st.cache_data(show_spinner="Scoring high-burnout classifiers on held-out data...")
def is_burnout_metrics() -> dict:
    df = load_raw_data()
    copy2 = build_classification_frame(df)
    train_data = copy2[CLASSIFICATION_PREDICTORS + ["Pre_Semester_GPA"]]
    target = copy2["Is_Burn_Out"]

    assets = load_is_burnout_assets()
    scaler = assets.scaler
    feat_names = list(scaler.feature_names_in_)
    scaled = pd.DataFrame(scaler.transform(train_data[feat_names]), columns=feat_names, index=train_data.index)

    _, x_test, _, y_test = train_test_split(scaled, target, stratify=target, test_size=0.4, random_state=26)

    exog_names = assets.log_model.model.exog_names
    x_test_sm = sm.add_constant(x_test, has_constant="add").reindex(columns=exog_names)
    log_prob = assets.log_model.predict(x_test_sm)
    log_auc = roc_auc_score(y_test, log_prob)

    tree_prob = assets.tree_model.predict_proba(x_test.reindex(columns=assets.tree_model.feature_names_in_))[:, 1]
    tree_auc = roc_auc_score(y_test, tree_prob)

    return {"log_auc": log_auc, "tree_auc": tree_auc, "n_test": len(y_test)}

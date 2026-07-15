"""
Loads every pickled model and scaler exactly once per running app process.

st.cache_resource keeps the deserialized objects in memory across reruns and
across users, so no model is ever re-trained or re-loaded from disk on every
click — inference on new inputs is the only thing that happens interactively.
"""

from dataclasses import dataclass

import joblib
import statsmodels.api as sm
import streamlit as st

REGRESSION_MODEL_PATH = "models/Regression/ols_model.pickle"
REGRESSION_SCALER_PATH = "models/Regression/scaler.joblib"

BURNOUT_LOG_MODEL_PATH = "models/Classification/Burnout_Risk_Level/burnout_risk_level_log_model.pickle"
BURNOUT_TREE_MODEL_PATH = "models/Classification/Burnout_Risk_Level/burnout_risk_level_tree_model.pickle"
BURNOUT_SCALER_PATH = "models/Classification/Burnout_Risk_Level/burnout_risk_level_scaler.joblib"

IS_BURNOUT_LOG_MODEL_PATH = "models/Classification/Is_Burnout/is_burnout_log_model.pickle"
IS_BURNOUT_TREE_GRID_PATH = "models/Classification/Is_Burnout/is_burnout_decision_tree_grid.pkl"
IS_BURNOUT_SCALER_PATH = "models/Classification/Is_Burnout/is_burnout_scaler.joblib"

IS_BURNOUT_DEFAULT_THRESHOLD = 0.2579089047156701


@dataclass
class RegressionAssets:
    model: object
    scaler: object


@dataclass
class BurnoutLevelAssets:
    log_model: object
    tree_model: object
    scaler: object


@dataclass
class IsBurnoutAssets:
    log_model: object
    tree_model: object
    tree_threshold: float
    tree_cv_auc: float
    scaler: object


@st.cache_resource(show_spinner="Loading regression model...")
def load_regression_assets() -> RegressionAssets:
    model = sm.load(REGRESSION_MODEL_PATH)
    scaler = joblib.load(REGRESSION_SCALER_PATH)
    return RegressionAssets(model=model, scaler=scaler)


@st.cache_resource(show_spinner="Loading burnout-risk classifiers...")
def load_burnout_level_assets() -> BurnoutLevelAssets:
    log_model = sm.load(BURNOUT_LOG_MODEL_PATH)
    tree_model = joblib.load(BURNOUT_TREE_MODEL_PATH)
    scaler = joblib.load(BURNOUT_SCALER_PATH)
    return BurnoutLevelAssets(log_model=log_model, tree_model=tree_model, scaler=scaler)


@st.cache_resource(show_spinner="Loading high-burnout classifiers...")
def load_is_burnout_assets() -> IsBurnoutAssets:
    log_model = sm.load(IS_BURNOUT_LOG_MODEL_PATH)
    grid = joblib.load(IS_BURNOUT_TREE_GRID_PATH)
    scaler = joblib.load(IS_BURNOUT_SCALER_PATH)
    return IsBurnoutAssets(
        log_model=log_model,
        tree_model=grid["model"],
        tree_threshold=grid.get("threshold", 0.5),
        tree_cv_auc=grid.get("best_cv_score", float("nan")),
        scaler=scaler,
    )

"""
Turns raw form inputs into predictions. Every function here only calls
.predict()/.predict_proba() on already-trained, already-loaded models —
nothing is fit here except a trivial pandas reindex.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lib.data import (
    BURNOUT_MAP_INV,
    engineer_classification_row,
    engineer_regression_row,
)
from lib.models import BurnoutLevelAssets, IsBurnoutAssets, RegressionAssets


def predict_gpa(raw_inputs: dict, assets: RegressionAssets, ci_alpha: float = 0.04):
    row = engineer_regression_row(raw_inputs)

    # Reproduce the notebook's mean-centered interaction term using the means
    # StandardScaler already learned for these two raw columns at train time.
    scaler = assets.scaler
    feat_names = list(scaler.feature_names_in_)
    mean_wgh = scaler.mean_[feat_names.index("Weekly_GenAI_Hours")]
    mean_pad = scaler.mean_[feat_names.index("Perceived_AI_Dependency")]
    row["Depending_Hours"] = (row["Weekly_GenAI_Hours"] - mean_wgh) * (
        row["Perceived_AI_Dependency"] - mean_pad
    )

    row = row.reindex(columns=feat_names)
    scaled = pd.DataFrame(scaler.transform(row), columns=feat_names)

    exog_names = assets.model.model.exog_names  # includes 'const'
    scaled_sm = sm.add_constant(scaled, has_constant="add").reindex(columns=exog_names)

    pred_result = assets.model.get_prediction(scaled_sm)
    summary = pred_result.summary_frame(alpha=ci_alpha)
    point = float(summary["mean"].iloc[0])
    # obs_ci_* is the prediction interval for a single new student (incorporates
    # residual variance), which is far more informative than the mean-response CI
    # once sample size is large — the latter shrinks to almost nothing at n≈30,000.
    ci_low, ci_high = float(summary["obs_ci_lower"].iloc[0]), float(summary["obs_ci_upper"].iloc[0])

    # GPA is bounded on a 0-4 scale in this dataset
    point_clipped = float(np.clip(point, 0, 4))
    return {
        "prediction": point_clipped,
        "raw_prediction": point,
        "ci_low": float(np.clip(ci_low, 0, 4)),
        "ci_high": float(np.clip(ci_high, 0, 4)),
        "gpa_change": point_clipped - raw_inputs["Pre_Semester_GPA"],
    }


def predict_burnout_level(raw_inputs: dict, assets: BurnoutLevelAssets, method: str):
    feats = engineer_classification_row(raw_inputs)
    scaler = assets.scaler
    feat_names = list(scaler.feature_names_in_)
    row = pd.DataFrame([feats]).reindex(columns=feat_names)
    scaled = pd.DataFrame(scaler.transform(row), columns=feat_names)

    if method == "Multinomial Logistic Regression":
        exog_names = assets.log_model.model.exog_names
        scaled_sm = sm.add_constant(scaled, has_constant="add").reindex(columns=exog_names)
        probs = assets.log_model.predict(scaled_sm).iloc[0]
        # statsmodels MNLogit orders categories 1..J as columns 0..J-1
        class_probs = {BURNOUT_MAP_INV[i + 1]: float(p) for i, p in enumerate(probs)}
        pred_class = max(class_probs, key=class_probs.get)
    else:
        tree = assets.tree_model
        scaled_ordered = scaled.reindex(columns=tree.feature_names_in_)
        proba = tree.predict_proba(scaled_ordered)[0]
        class_probs = {BURNOUT_MAP_INV[c]: float(p) for c, p in zip(tree.classes_, proba)}
        pred_class = BURNOUT_MAP_INV[int(tree.predict(scaled_ordered)[0])]

    return {"pred_class": pred_class, "probabilities": class_probs}


def predict_is_burnout(raw_inputs: dict, assets: IsBurnoutAssets, method: str):
    feats = engineer_classification_row(raw_inputs)
    scaler = assets.scaler
    feat_names = list(scaler.feature_names_in_)
    row = pd.DataFrame([feats]).reindex(columns=feat_names)
    scaled = pd.DataFrame(scaler.transform(row), columns=feat_names)

    if method == "Logistic Regression":
        exog_names = assets.log_model.model.exog_names
        scaled_sm = sm.add_constant(scaled, has_constant="add").reindex(columns=exog_names)
        prob = float(assets.log_model.predict(scaled_sm).iloc[0])
        threshold = 0.5
    else:
        tree = assets.tree_model
        scaled_ordered = scaled.reindex(columns=tree.feature_names_in_)
        prob = float(tree.predict_proba(scaled_ordered)[0][1])
        threshold = assets.tree_threshold

    return {
        "probability": prob,
        "threshold": threshold,
        "flag": prob >= threshold,
    }

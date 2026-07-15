"""
Data loading + feature engineering.

Every transformation here is a 1:1 mirror of the steps performed in
AI_Students.ipynb, so that predictions made through the app line up
exactly with what the notebook's models were trained on.

Nothing in this module fits or trains a predictive model. The only
"fitting" that ever happens (StandardScaler) is precomputed offline and
shipped as .joblib files in /models — see lib/models.py.
"""

import pandas as pd
import streamlit as st

DATA_PATH = "ai_student_impact_dataset.csv"

NOMINAL_COLS = ["Major_Category", "Primary_Use_Case", "Institutional_Policy"]

YEAR_MAP = {"Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4, "Graduate": 5}
SKILL_MAP = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
BURNOUT_MAP = {"Low": 1, "Medium": 2, "High": 3}
BURNOUT_MAP_INV = {v: k for k, v in BURNOUT_MAP.items()}

REG_DROPPED_DUMMIES = [
    "Major_Category_Arts",
    "Primary_Use_Case_Ideation",
    "Institutional_Policy_Actively_Encouraged",
]

REGRESSION_TARGET = "Post_Semester_GPA"

REGRESSION_DROP_FOR_TRAIN = [
    "Tool_Diversity", "Burnout_Risk_Level",
    "Major_Category_Business", "Major_Category_Humanities",
    "Major_Category_Medical", "Major_Category_STEM",
    "Primary_Use_Case_Copywriting/Drafting", "Primary_Use_Case_Summarizing_Reading",
    REGRESSION_TARGET, "Institutional_Policy_Allowed_With_Citation",
    "Paid_Subscription", "Institutional_Policy_Strict_Ban", "Skill_Retention_Score",
]

CLASSIFICATION_PREDICTORS = [
    "Anxiety_Level_During_Exams", "Year_of_Study", "Tool_Diversity",
    "AI_Engagement", "AI_Dependency_Ratio", "Total_Study",
    "Perceived_AI_Dependency", "Skill_Retention_Score",
]


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def build_encoded_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors notebook cells 26-34: dummy-encode, ordinal-map, drop outliers."""
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    new_df = df.copy()
    new_df.drop(columns=["Student_ID"], inplace=True)
    new_df = pd.get_dummies(new_df, columns=NOMINAL_COLS, dtype=int)
    new_df[bool_cols] = new_df[bool_cols].astype(int)

    new_df["Year_of_Study"] = new_df["Year_of_Study"].map(YEAR_MAP)
    new_df["Prompt_Engineering_Skill"] = new_df["Prompt_Engineering_Skill"].map(SKILL_MAP)
    new_df["Burnout_Risk_Level"] = new_df["Burnout_Risk_Level"].map(BURNOUT_MAP)

    # drop the same outlier rows the notebook drops before modelling
    new_df = new_df.drop(index=df.loc[df["Traditional_Study_Hours"] > 32].index)
    return new_df


@st.cache_data(show_spinner=False)
def build_classification_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors notebook cells 66-68: engineered features for burnout models."""
    new_df = build_encoded_frame(df)
    copy2 = new_df.drop(columns=REG_DROPPED_DUMMIES)

    copy2["AI_Engagement"] = (
        copy2["Weekly_GenAI_Hours"] * copy2["Tool_Diversity"] * copy2["Prompt_Engineering_Skill"]
    )
    copy2["AI_Dependency_Ratio"] = copy2["Weekly_GenAI_Hours"] / (copy2["Traditional_Study_Hours"] + 1)
    copy2["Total_Study"] = copy2["Weekly_GenAI_Hours"] + copy2["Traditional_Study_Hours"]
    copy2["GPA_Change"] = copy2["Post_Semester_GPA"] - copy2["Pre_Semester_GPA"]
    copy2["Is_Burn_Out"] = (copy2["Burnout_Risk_Level"] == 3).astype(int)
    return copy2


def engineer_regression_row(raw: dict) -> pd.DataFrame:
    """Turn a single raw user input dict into the exact feature row the OLS model expects."""
    row = {
        "Year_of_Study": YEAR_MAP[raw["Year_of_Study"]],
        "Pre_Semester_GPA": raw["Pre_Semester_GPA"],
        "Weekly_GenAI_Hours": raw["Weekly_GenAI_Hours"],
        "Prompt_Engineering_Skill": SKILL_MAP[raw["Prompt_Engineering_Skill"]],
        "Traditional_Study_Hours": raw["Traditional_Study_Hours"],
        "Perceived_AI_Dependency": raw["Perceived_AI_Dependency"],
        "Anxiety_Level_During_Exams": raw["Anxiety_Level_During_Exams"],
        "Primary_Use_Case_Debugging/Troubleshooting": int(raw["Primary_Use_Case"] == "Debugging/Troubleshooting"),
        "Primary_Use_Case_Direct_Answer_Generation": int(raw["Primary_Use_Case"] == "Direct_Answer_Generation"),
    }
    # Depending_Hours = (Weekly_GenAI_Hours - train_mean) * (Perceived_AI_Dependency - train_mean)
    # is an interaction term centered on the *training set* means. It is filled in by
    # lib/inference.py (which has access to those means via the fitted scaler) right
    # before scaling, so it is intentionally left out of this row.
    return pd.DataFrame([row])


def engineer_classification_row(raw: dict) -> dict:
    """Compute the engineered predictors shared by both burnout models."""
    weekly_ai = raw["Weekly_GenAI_Hours"]
    tool_div = raw["Tool_Diversity"]
    skill = SKILL_MAP[raw["Prompt_Engineering_Skill"]]
    trad_hours = raw["Traditional_Study_Hours"]

    return {
        "Anxiety_Level_During_Exams": raw["Anxiety_Level_During_Exams"],
        "Year_of_Study": YEAR_MAP[raw["Year_of_Study"]],
        "Tool_Diversity": tool_div,
        "AI_Engagement": weekly_ai * tool_div * skill,
        "AI_Dependency_Ratio": weekly_ai / (trad_hours + 1),
        "Total_Study": weekly_ai + trad_hours,
        "Perceived_AI_Dependency": raw["Perceived_AI_Dependency"],
        "Skill_Retention_Score": raw["Skill_Retention_Score"],
        "Pre_Semester_GPA": raw.get("Pre_Semester_GPA"),
    }

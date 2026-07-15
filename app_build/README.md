# AI Impact on Students — Prediction Dashboard

An interactive Streamlit app built on top of `AI_Students.ipynb`. It lets you
explore the dataset, see the exact feature-engineering pipeline used before
training, and run **live inference** with the models already saved in
`models/` — the app never trains or re-fits any predictive model.

## What's inside

| Page | What it does |
|---|---|
| **Overview** | Dataset snapshot and key stats |
| **Explore the Data** | Interactive distributions, outlier check, category counts, correlation heatmap |
| **Feature Engineering** | Explains encoding/scaling/engineered features, shows correlation-with-target and VIF tables |
| **Predict: Academic Performance** | OLS linear regression → predicted `Post_Semester_GPA` with a prediction interval |
| **Predict: Burnout Risk** | Two classifiers: 3-level `Burnout_Risk_Level` (Multinomial Logit / Decision Tree) and binary `Is_Burn_Out` (Logistic Regression / tuned Decision Tree) |
| **Model Cards** | Held-out test metrics for every model (computed once, cached, never retrained) |

## Project layout

```
app.py                     # entry point + sidebar navigation
lib/
  data.py                  # feature engineering, mirrors the notebook exactly
  models.py                # cached model/scaler loading (st.cache_resource)
  inference.py             # turns form inputs into predictions
  evaluation.py            # cached held-out metrics for the Model Cards page
  style.py                 # custom CSS + UI components
pages_content/
  overview.py, eda.py, feature_eng.py,
  predict_regression.py, predict_burnout.py, model_cards.py
models/                    # your existing pickled models (unchanged)
  Regression/ols_model.pickle, scaler.joblib
  Classification/Burnout_Risk_Level/*.pickle, burnout_risk_level_scaler.joblib  *(new)*
  Classification/Is_Burnout/*.pickle, *.pkl, is_burnout_scaler.joblib          *(new)*
ai_student_impact_dataset.csv
requirements.txt
.streamlit/config.toml     # dark theme
```

### Two new files you didn't have before

Your notebook never saved a `StandardScaler` for the two burnout classifiers
(only for the regression model). Since those classifiers need input scaled
the same way they were trained, this package includes two small pre-fitted
scaler files, generated once from your dataset with the same code the
notebook uses:

- `models/Classification/Burnout_Risk_Level/burnout_risk_level_scaler.joblib`
- `models/Classification/Is_Burnout/is_burnout_scaler.joblib`

They're loaded read-only, just like your other model files — nothing is
fit at runtime.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (share.streamlit.io)

1. Commit everything in this folder into the root of your
   `AI_Students_Impact` GitHub repo (alongside your existing `models/`
   folder and `ai_student_impact_dataset.csv` — the app expects those
   relative paths).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**.
3. Pick your repo/branch and set **Main file path** to `app.py`.
4. Deploy. First load will take a little longer while models are cached;
   every load after that is fast since Streamlit keeps cached resources
   warm between reruns.

## Why nothing gets retrained

- All models/scalers are loaded from disk exactly once via
  `@st.cache_resource` in `lib/models.py`.
- The two new scalers were fit **once, offline**, and saved as `.joblib` —
  not fit inside the app.
- The only "fitting" that happens at runtime is a `pandas.DataFrame.reindex`
  and `scaler.transform()` call on a single-row input — pure inference.
- Held-out metrics on the **Model Cards** page reuse a deterministic
  `train_test_split` (same `random_state` as the notebook) purely to score
  already-trained models on unseen rows, then cache the result with
  `@st.cache_data` so it only runs once per deployment.

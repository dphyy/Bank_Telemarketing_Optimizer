# 📁 Code — Module Reference

This folder contains every Python module that powers the Bank Telemarketing Optimizer. Below is a detailed breakdown of each file's role, the features it implements, and how the modules relate to one another.

---

## Module Map

```
Code/
├── app.py                  ← Streamlit dashboard (entry point)
├── chatbot.py              ← AI assistant (HuggingFace-powered)
├── dashboard_pipeline.py   ← Core ML pipeline & business logic
├── dashboard_visuals.py    ← Chart builders (Plotly + Matplotlib)
├── evaluation.py           ← Custom asymmetric F-beta scorer
├── ml.py                   ← Model definitions & training wrappers
├── preprocessing.py        ← Data cleaning & feature engineering
└── train_dashboard_models.py ← CLI pre-training script
```

---

## `preprocessing.py`

**Purpose:** Ingest raw CSV data, clean it, engineer features, and produce train/test splits ready for modelling.

### Features

| Feature | Detail |
|---|---|
| **Schema enforcement** | Ensures all expected raw columns are present; fills missing ones with dataset-derived medians/modes via `_reference_raw_defaults()` |
| **Categorical imputation** | `job`, `marital`, `housing`, `loan` imputed with column mode |
| **Numeric imputation** | `age`, `campaign`, `euribor3m`, etc. imputed with column median |
| **Ordinal education encoding** | Maps education levels to integers (1 = illiterate → 7 = university degree) respecting domain ordering |
| **`contacted` feature engineering** | Derives a binary flag from `pdays` (999 = never contacted → 0, any other value → 1) |
| **Column dropping** | Drops `duration` (data leakage), `pdays` (replaced by `contacted`), `emp.var.rate` (collinear with `euribor3m`/`nr.employed`), and `default` (near-zero variance) |
| **ColumnTransformer pipeline** | `RobustScaler` for skewed features (`age`, `campaign`, `previous`); `StandardScaler` for symmetric/bounded features; `OneHotEncoder` for categoricals; passthrough for `contacted` |
| **Stratified train/test split** | 70/30 split with `stratify=y` to preserve class distribution |
| **SMOTE oversampling** | Applied to **training data only** to address class imbalance; test set is never resampled |

### Key Exports

- `preprocess_data()` — full pipeline returning `(X_train, y_train, X_train_res, y_train_res, X_test, y_test, raw_train_frame, raw_test_frame)`
- `clean_raw_data()` — standalone cleaner for inference-time single-row inputs
- `build_preprocessor()` — returns an unfitted `ColumnTransformer`
- `load_data()` — CSV loader with path resolution

---

## `evaluation.py`

**Purpose:** Define the custom business-driven metric used throughout training and model selection.

### Features

| Feature | Detail |
|---|---|
| **Asymmetric F-beta scorer** | `custom_scorer(y_test, y_pred, cost_fp, cost_fn)` computes F-beta where `beta = √(cost_fn / cost_fp)`. With the default `cost_fn=4, cost_fp=1`, beta ≈ 2, penalising missed subscriptions more heavily than wasted calls |

---

## `ml.py`

**Purpose:** Define, tune, and train all classifiers. Each function is independent — callers pass in data; nothing is loaded at import time.

### Models Implemented

| Model | Variants | Imbalance Strategy |
|---|---|---|
| Logistic Regression | Balanced weights / SMOTE-resampled | `class_weight='balanced'` or SMOTE |
| K-Nearest Neighbours | Balanced weights / SMOTE-resampled | `class_weight='balanced'` or SMOTE |
| Decision Tree | Balanced weights / SMOTE-resampled | `class_weight='balanced'` or SMOTE |
| Random Forest | Balanced weights / SMOTE-resampled | `class_weight='balanced'` or SMOTE |
| XGBoost | Balanced (`scale_pos_weight`) / SMOTE | `scale_pos_weight` or SMOTE |
| LightGBM | Balanced (`is_unbalance`) / SMOTE | `is_unbalance=True` or SMOTE |
| MLP Neural Network | SMOTE-resampled | SMOTE |
| Voting Ensemble | Soft-voting over top models | Inherited from members |

### Features

| Feature | Detail |
|---|---|
| **GridSearchCV hyperparameter tuning** | F-beta-scored `StratifiedKFold` cross-validation for applicable models (KNN, Decision Tree, Random Forest, XGBoost, LightGBM, MLP) |
| **SHAP value computation** | `compute_shap_values(model, X)` dispatches to `TreeExplainer` (tree models) or `KernelExplainer` (others) |
| **Global estimator registry** | `BEST_ESTIMATORS` dict stores every fitted model by name for downstream use |
| **Decoupled evaluate / fit** | `evaluate_fbeta()` scores an already-fitted model; does not refit, preventing test-set leakage |

---

## `dashboard_pipeline.py`

**Purpose:** The central orchestration layer. Bridges raw data, trained models, and the business profit framework for the dashboard.

### Key Data Structures

- **`ModelRun`** — dataclass holding a fitted estimator, training status, and metadata
- **`TrainingBundle`** — dataclass holding all models, test data, probabilities, and feature names
- **`FittedModelBundle`** — wraps a `(preprocessor, estimator)` pair; exposes `predict` / `predict_proba` that accept raw (un-preprocessed) DataFrames

### Features

| Feature | Detail |
|---|---|
| **`build_training_bundle()`** | Orchestrates the full train–evaluate cycle; optionally loads pre-saved artefacts from `Models/` to skip retraining |
| **Profit-based threshold selection** | For each model, sweeps probability thresholds and selects the one maximising `TP × revenue − FP × call_cost` |
| **`evaluate_at_profits()`** | Returns per-model expected profit, confusion matrix values, precision, recall, and F-beta at the optimal threshold |
| **`recommended_model_row()`** | Identifies the single best model by expected profit |
| **`focused_comparison_rows()`** | Assembles a comparison table including profit gains vs. the "Call Everyone" baseline and vs. Logistic Regression |
| **`cost_sensitivity()` / `cost_sensitivity_by_model()`** | Sweep revenue-to-cost ratios and compute expected profit at each ratio — used for the sensitivity analysis charts |
| **`score_customers()`** | Scores an arbitrary batch of raw customer rows and returns ranked subscription probabilities |
| **`shap_explanation()` / `probability_shap_explanation()`** | Compute global SHAP summaries or single-customer waterfall explanations |
| **`input_field_hints()`** | Returns display metadata (type, options, ranges) for every input field, used to dynamically build the dashboard's input form |
| **artifact persistence** | Saves and loads fitted models + metadata via `joblib` / JSON, with a version flag to invalidate stale artefacts |

---

## `dashboard_visuals.py`

**Purpose:** All chart and plot generation, keeping visualisation concerns separate from business logic.

### Charts

| Function | Chart Type | Description |
|---|---|---|
| `expected_profit_by_model()` | Plotly bar | Ranks all models by expected profit |
| `confusion_matrix_figure()` | Plotly heatmap | Confusion matrix for the best model |
| `focused_model_comparison()` | Plotly bar | Compares selected models: expected profit, gain vs. baselines, and F-beta |
| `model_ratio_curves()` | Plotly line | Expected profit vs. revenue-to-cost ratio for all models (sensitivity curves) |
| `render_shap_summary_plot()` | Matplotlib / SHAP | Beeswarm summary of global feature importances |
| `render_shap_dependence_plot()` | Matplotlib / SHAP | Feature dependence plot; x-axis relabelled to raw-scale values for readability |
| `render_probability_waterfall_plot()` | Matplotlib / SHAP | Single-customer probability waterfall; labels rewritten to human-readable raw values |

### Theming

A consistent colour palette (`PALETTE`) and `style_plot()` helper are applied to every Plotly figure for a unified dark-navy-on-white visual identity.

---

## `app.py`

**Purpose:** The Streamlit dashboard — the primary user interface.

### Dashboard Tabs & Features

| Tab | Features |
|---|---|
| **Model Performance** | KPI metrics (best model, expected profit, F-beta, precision, recall); profit bar chart across all models; confusion matrix; focused model comparison chart |
| **Sensitivity Analysis** | Revenue-to-cost ratio curves showing where each model outperforms the call-everyone baseline |
| **Score Customers** | Upload a CSV of raw customer data; receive a ranked list with subscription probabilities and recommended actions |
| **Single Customer** | Manual input form for all 19 raw features; returns probability, SHAP waterfall explanation, and call/skip recommendation |
| **Explainability** | Global SHAP summary plot and SHAP dependence plots for any feature, per model |
| **AI Chatbot** | Contextual Q&A assistant scoped to the project (see `chatbot.py`) |

### Technical Details

- Uses **background threading** (`ThreadPoolExecutor`) so model training does not block the UI
- Dashboard state is managed with `st.session_state`; models are trained or loaded once and cached
- All input fields are dynamically generated from `input_field_hints()`, so adding a new feature requires no UI changes

---

## `chatbot.py`

**Purpose:** An AI assistant embedded in the dashboard, scoped exclusively to project-related questions.

### Features

| Feature | Detail |
|---|---|
| **HuggingFace Inference API** | Uses `huggingface_hub.InferenceClient`; compatible with both legacy and OpenAI-compatible API (>=0.26) |
| **Default model** | `Qwen/Qwen2.5-7B-Instruct` (configurable via `HF_CHAT_MODEL`) |
| **Out-of-scope guard** | Keyword + multi-word phrase matching rejects questions unrelated to the project before hitting the API |
| **Context injection** | `build_chat_context()` prepends a structured system prompt containing the current best model, profit metrics, threshold, and top SHAP features, giving the LLM grounding in the current session's results |
| **Token budget management** | Conversation history is trimmed to stay within `MAX_TOTAL_TOKENS` (default 1800) for the free API tier |

---

## `train_dashboard_models.py`

**Purpose:** A standalone CLI script to pre-train all models and save artefacts to `Models/`, separating the training workload from dashboard startup.

### Usage

```bash
python Code/train_dashboard_models.py
```

Prints a summary of which models saved successfully and which (if any) failed with errors.

---

## Data Flow Summary

```
bank_telemarketing.csv
        │
        ▼
  preprocessing.py          ← clean, encode, scale, SMOTE
        │
        ▼
     ml.py                  ← train models, tune hyperparameters
        │
        ▼
 dashboard_pipeline.py      ← threshold optimisation, profit evaluation,
        │                      SHAP computation, artifact persistence
        │
   ┌────┴──────────────────────────────┐
   ▼                                   ▼
app.py  ←── dashboard_visuals.py    chatbot.py
(Streamlit UI)   (Plotly/SHAP charts)  (AI assistant)
```

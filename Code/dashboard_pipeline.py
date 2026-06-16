# Import necessary libraries and modules
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import preprocessing as prep
from evaluation import custom_scorer
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
import ml as ml_module



# Initial configuration and constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "bank_telemarketing.csv"
MODEL_DIR = PROJECT_ROOT / "Models"
METADATA_PATH = MODEL_DIR / "dashboard_model_metadata.json"
RANDOM_STATE = 42
ARTIFACT_VERSION = 1    # increase if we change the training pipeline in a way that invalidates old artifacts
TARGET_COLUMN = "y"

# RAW_FEATURES intentionally excludes columns that are always dropped during
# cleaning (duration, pdays, emp.var.rate, default) so that downstream callers
# only surface fields that survive into the model.
RAW_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "previous",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

# Display-only: the full set of raw columns accepted from external input before
# cleaning (mirrors prep.RAW_INPUT_DEFAULTS).  Used by input_field_hints and
# _complete_raw_input_frame only.
_ALL_INPUT_FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]


@dataclass
class ModelRun:
    name: str
    estimator: BaseEstimator | None
    status: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class TrainingBundle:
    # raw_test_frame holds the *original* pre-clean test rows returned by
    # preprocess_data.  It is used as input for SHAP and feature-importance so
    # that the preprocessor inside FittedModelBundle can transform it fresh.
    raw_test_frame: pd.DataFrame
    X_test: pd.DataFrame              # preprocessed test features (for probabilities / metrics)
    y_train: pd.Series
    y_test: pd.Series
    models: dict[str, ModelRun]
    probabilities: dict[str, np.ndarray]        # test-set positive-class probabilities
    train_probabilities: dict[str, np.ndarray]  # train-set probabilities (threshold selection only)
    feature_names: list[str]
    unavailable_optional: list[str]


@dataclass
class FittedModelBundle:
    """Pairs a fitted ColumnTransformer preprocessor with a fitted estimator.
 
    predict / predict_proba / decision_function all accept *raw* (pre-clean)
    DataFrames.  The bundle cleans then transforms internally so callers never
    have to pre-process manually.
    """
    preprocess: Any   # fitted ColumnTransformer
    model: Any        # fitted sklearn estimator
 
    @property
    def named_steps(self) -> dict[str, Any]:
        return {"preprocess": self.preprocess, "model": self.model}
 
    def _prepare(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Clean raw input and apply the fitted preprocessor.
 
        The ColumnTransformer's .transform() returns a bare numpy array, which
        would trigger 'X does not have valid feature names' warnings when fed
        to estimators fitted on a named DataFrame. Re-wrap the array using the
        preprocessor's own output feature names so column names are preserved
        end-to-end.
        """
        if isinstance(X, pd.DataFrame):
            X = prep.clean_raw_data(X.copy())
            index = X.index
        else:
            index = None
        transformed = self.preprocess.transform(X)
        return pd.DataFrame(
            transformed,
            columns=self.preprocess.get_feature_names_out(),
            index=index,
        )
    
    def _prepare_preprocessed(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Pass through already-preprocessed input unchanged.
 
        Used by predict_model_probabilities where X has already been through
        preprocess_data (clean + transform) and must not be processed again.
        Keeping the DataFrame (with its column names) avoids
        'X does not have valid feature names' warnings from estimators that
        were fitted on a DataFrame with named columns.
        """
        return X
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(self._prepare(X))
 
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._prepare(X))
 
    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.decision_function(self._prepare(X))



# Model specs
# Each entry: (display_name, ml_module_function_name, BEST_ESTIMATORS_key)
_MODEL_SPECS: list[tuple[str, str, str]] = [
    ("Logistic Regression",         "train_logistic_regression_balanced",  "LogisticRegression_balanced"),
    ("K-Nearest Neighbors",         "train_knn_balanced",                  "KNN_balanced"),
    ("Decision Tree",               "train_decision_tree_balanced",        "DecisionTree_balanced"),
    ("Random Forest",               "train_random_forest_balanced",        "RandomForest_balanced"),
    ("Neural Network",              "train_mlp_balanced",                  "MLP_balanced"),
    ("XGBoost",                     "train_xgboost_balanced",              "XGBoost_balanced"),
    ("LightGBM",                    "train_lightgbm_balanced",             "LightGBM_balanced"),
    ("Voting LR + RF + MLP",        "train_voting_lr_rf_mlp_balanced",     "Voting_LR_RF_MLP_balanced"),
    ("Voting LR + RF + XGBoost",    "train_voting_lr_rf_xgb_balanced",     "Voting_LR_RF_XGB_balanced"),
    ("Voting LR + RF + LightGBM",   "train_voting_lr_rf_lgbm_balanced",    "Voting_LR_RF_LGBM_balanced"),
]

EXPECTED_MODEL_NAMES: set[str] = {name for name, _, _ in _MODEL_SPECS}



# Training pipeline
def train_candidate_models(
    file_path: str | Path = DATA_PATH,
) -> tuple[dict[str, ModelRun], list[str]]:
    """Run the full preprocessing pipeline then train every model in ml.py.

    Flow
    ----
    1. ``prep.preprocess_data`` loads the CSV, splits 70/30 train/test on raw rows,
       cleans, fits the ColumnTransformer on the training split only (no leakage),
       applies SMOTE to the training data only, and returns both the plain
       preprocessed arrays and the SMOTE-resampled ones.
    2. Each ml.py training wrapper receives the appropriate preprocessed splits
       and writes its best estimator into ``ml.BEST_ESTIMATORS``.
    3. We retrieve that estimator and pair it with the *same* fitted preprocessor
       (extracted from X_train's pipeline) inside a ``FittedModelBundle`` so
       inference is self-contained and leak-free.
    """
    (
        X_tr,       # preprocessed, no SMOTE
        y_tr,
        X_train_res,   # preprocessed + SMOTE
        y_train_res,
        X_te,        # preprocessed
        y_te,
        _raw_train_frame,
        _raw_test_frame,
    ) = prep.preprocess_data(file_path)

    # Re-fit a fresh preprocessor on the training split only to avoid any
    # test-set leakage in the bundle's preprocessor.
    X_train_raw_clean = prep.clean_raw_data(_raw_train_frame.copy())
    preprocessor = prep.build_preprocessor()
    preprocessor.fit(X_train_raw_clean)

    models: dict[str, ModelRun] = {}
    unavailable: list[str] = []

    for display_name, fn_name, best_key in _MODEL_SPECS:
        fn = getattr(ml_module, fn_name, None)
        if fn is None:
            unavailable.append(display_name)
            models[display_name] = ModelRun(
                name=display_name, estimator=None,
                status="unavailable", error=f"{fn_name} not found in ml module",
            )
            continue
        try:
            fn(X_tr, y_tr, X_te, y_te)
            fitted_model = ml_module.BEST_ESTIMATORS.get(best_key)
            if fitted_model is None:
                raise RuntimeError(
                    f"ml.BEST_ESTIMATORS['{best_key}'] was not set by {fn_name}"
                )
            bundle = FittedModelBundle(preprocess=preprocessor, model=fitted_model)
            models[display_name] = ModelRun(
                name=display_name,
                estimator=bundle,
                status="ok",
                metadata={
                    "model_type": "ensemble" if display_name.startswith("Voting") else "base",
                    "tuned": True,
                    "tuning_metric": "F-Beta",
                },
            )
        except Exception as exc:
            models[display_name] = ModelRun(
                name=display_name, estimator=None,
                status="failed", error=str(exc),
            )

    return models, unavailable



# Probability helpers
def _positive_probabilities_preprocessed(model: Any, X_preprocessed: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Get positive-class probabilities from a FittedModelBundle using already-preprocessed input.

    Bypasses the bundle's internal clean+transform so that X_test (which has
    already been through preprocess_data) is not processed a second time.
    """
    if isinstance(model, FittedModelBundle):
        arr = model._prepare_preprocessed(X_preprocessed)
        inner = model.model
    else:
        # Fallback for plain sklearn estimators (should not occur at runtime).
        arr = X_preprocessed.to_numpy() if isinstance(X_preprocessed, pd.DataFrame) else np.asarray(X_preprocessed)
        inner = model
    if hasattr(inner, "predict_proba"):
        return inner.predict_proba(arr)[:, 1]
    if hasattr(inner, "decision_function"):
        raw = inner.decision_function(arr)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(inner.predict(arr), dtype=float)


def predict_model_probabilities(
    models: dict[str, ModelRun],
    X_test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Compute positive-class probabilities on the preprocessed test set.

    X_test has already been through preprocess_data (clean + transform), so we
    bypass the bundle's internal preprocessing and feed the array directly to
    the underlying model.
    """
    return {
        name: _positive_probabilities_preprocessed(run.estimator, X_test)
        for name, run in models.items()
        if run.status == "ok" and run.estimator is not None
    }



# Caching models for faster dashboard loading
def _artifact_path(model_name: str) -> Path:
    safe = model_name.lower().replace("+", "plus").replace(" ", "_").replace("/", "_")
    return MODEL_DIR / f"{safe}.joblib"
 
 
def save_model_artifacts(
    models: dict[str, ModelRun],
    probabilities: dict[str, np.ndarray],
    train_probabilities: dict[str, np.ndarray],
    feature_names: list[str],
    y_test: pd.Series,
    unavailable_optional: list[str],
    model_dir: str | Path = MODEL_DIR,
) -> None:
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "training_mode": "preprocessed_wrappers",
        "feature_names": feature_names,
        "unavailable_optional": unavailable_optional,
        "models": {},
    }
    for name, run in models.items():
        model_meta: dict[str, Any] = {
            "status": run.status,
            "error": run.error,
            **(run.metadata or {}),
        }
        if run.status == "ok" and run.estimator is not None:
            path = _artifact_path(name)
            joblib.dump(
                {
                    "name": name,
                    "estimator": run.estimator,
                    "probabilities": probabilities.get(name),
                    "train_probabilities": train_probabilities.get(name),
                    "y_test": np.asarray(y_test),
                    "metadata": run.metadata or {},
                },
                path,
            )
            model_meta["artifact"] = str(path.relative_to(PROJECT_ROOT))
        metadata["models"][name] = model_meta
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
 
 
def load_model_artifacts(
    model_dir: str | Path = MODEL_DIR,
) -> tuple[dict[str, ModelRun], dict[str, np.ndarray], dict[str, np.ndarray], list[str], list[str]] | None:
    metadata_path = Path(model_dir) / METADATA_PATH.name
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("artifact_version") != ARTIFACT_VERSION
        or metadata.get("training_mode") != "preprocessed_wrappers"
    ):
        return None
    models: dict[str, ModelRun] = {}
    probabilities: dict[str, np.ndarray] = {}
    train_probabilities: dict[str, np.ndarray] = {}
    for name, meta in metadata.get("models", {}).items():
        artifact = meta.get("artifact")
        if meta.get("status") != "ok" or not artifact:
            models[name] = ModelRun(
                name=name, estimator=None,
                status=meta.get("status", "failed"),
                error=meta.get("error"),
            )
            continue
        loaded = joblib.load(PROJECT_ROOT / artifact)
        models[name] = ModelRun(
            name=name,
            estimator=loaded["estimator"],
            status="ok",
            metadata=loaded.get("metadata", {}),
        )
        if loaded.get("probabilities") is not None:
            probabilities[name] = np.asarray(loaded["probabilities"], dtype=float)
        if loaded.get("train_probabilities") is not None:
            train_probabilities[name] = np.asarray(loaded["train_probabilities"], dtype=float)
    return (
        models,
        probabilities,
        train_probabilities,
        list(metadata.get("feature_names", [])),
        list(metadata.get("unavailable_optional", [])),
    )



# Bundle construction (cache-aware entry point)
def build_training_bundle(
    file_path: str | Path = DATA_PATH,
    use_artifacts: bool = True,
) -> TrainingBundle:
    """Return a fully populated TrainingBundle.
 
    If ``use_artifacts`` is True and valid cached artifacts exist for all
    expected models, they are loaded directly.  Otherwise models are trained
    from scratch via the preprocessing → ml pipeline and saved for next time.
 
    Threshold selection uses train-set probabilities; all reported metrics are
    evaluated on the held-out test set.
    """
    # Always run preprocessing so we have splits for evaluation and SHAP
    # regardless of whether models come from cache.
    (
        X_train,
        y_train,
        _X_train_res,
        _y_train_res,
        X_test,
        y_test,
        raw_train_frame,
        raw_test_frame,
    ) = prep.preprocess_data(file_path)
    feature_names = list(X_test.columns)

    artifact_bundle = load_model_artifacts() if use_artifacts else None
    cache_valid = (
        artifact_bundle is not None
        and EXPECTED_MODEL_NAMES.issubset(set(artifact_bundle[0]))
    )

    if cache_valid:
        models, probabilities, train_probabilities, artifact_features, unavailable = artifact_bundle
        # Always recompute probabilities against the freshly-preprocessed test and training
        # set to guarantee alignment between probabilities and y_test.
        probabilities = predict_model_probabilities(
            {n: run for n, run in models.items() if run.status == "ok"},
            X_test,
        )
        train_probabilities = predict_model_probabilities(
            {n: run for n, run in models.items() if run.status == "ok"},
            X_train,
        )
        feature_names = artifact_features or feature_names
    else:
        models, unavailable = train_candidate_models(file_path)
        probabilities = predict_model_probabilities(models, X_test)
        train_probabilities = predict_model_probabilities(models, X_train)
        save_model_artifacts(models, probabilities, train_probabilities, feature_names, y_test, unavailable)

    # Reset the index on raw_test_frame so it is always 0-based.
    return TrainingBundle(
        raw_test_frame=raw_test_frame.reset_index(drop=True),
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models=models,
        probabilities=probabilities,
        train_probabilities=train_probabilities,
        feature_names=feature_names,
        unavailable_optional=unavailable,
    )



# Profit / threshold utilities
def expected_profit(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Expected profit = TP × (cost_fn − cost_fp) − FP × cost_fp.
 
    cost_fn is the revenue gained by correctly identifying a subscriber
    (avoided missed-opportunity); cost_fp is the wasted call cost incurred on
    a false positive.  Higher is better.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tp * (cost_fn - cost_fp) - fp * cost_fp)
 
 
def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> tuple[float, np.ndarray, float]:
    """Find the probability threshold that maximises expected profit.
 
    Must be called with *training-set* labels and probabilities.
    The returned threshold is then applied to the test set in _metrics_row
    so that threshold selection does not leak test-set information.
 
    Ties in profit are broken by choosing the threshold that makes the fewest
    calls (the most conservative decision boundary).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob_arr)
    sorted_prob = y_prob_arr[order]
    sorted_true = y_true_arr[order]
 
    positive_total = int(sorted_true.sum())
    negative_total = int(len(sorted_true) - positive_total)
    tp_cum = np.cumsum(sorted_true == 1)
    fp_cum = np.cumsum(sorted_true == 0)
 
    unique_end_idx = np.r_[
        np.where(sorted_prob[:-1] != sorted_prob[1:])[0],
        len(sorted_prob) - 1,
    ]
    thresholds = sorted_prob[unique_end_idx]
    tp = tp_cum[unique_end_idx]
    fp = fp_cum[unique_end_idx]
    # Profit at each candidate threshold.
    profits = tp * (cost_fn - cost_fp) - fp * cost_fp
    calls = tp + fp
 
    # Sentinel thresholds: call nobody → profit 0; call everyone → all
    # positives captured but all negatives also called.
    call_none_profit = 0.0
    call_everyone_profit = float(positive_total * (cost_fn - cost_fp) - negative_total * cost_fp)
    all_thresholds = np.r_[1.000001, thresholds, 0.0]
    all_profits = np.r_[call_none_profit, profits, call_everyone_profit]
    all_calls = np.r_[0, calls, len(y_true_arr)]
 
    max_profit = np.max(all_profits)
    tied = np.where(all_profits == max_profit)[0]
    # Among equally profitable thresholds, prefer the one that calls fewest.
    best_idx = tied[np.argmin(all_calls[tied])]
    best_threshold = float(all_thresholds[best_idx])
    best_pred = (y_prob_arr >= best_threshold).astype(int)
    return best_threshold, best_pred, float(all_profits[best_idx])
 
 
def _safe_auc(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")
 
 
def _metrics_row(
    name: str,
    y_true_train: pd.Series | np.ndarray,
    y_prob_train: np.ndarray,
    y_true_test: pd.Series | np.ndarray,
    y_prob_test: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, Any]:
    # Threshold selected on the training set to avoid test-set leakage.
    threshold, _, _ = find_best_threshold(y_true_train, y_prob_train, cost_fp, cost_fn)
    # All reported metrics are computed on the held-out test set using that threshold.
    y_pred_test = (np.asarray(y_prob_test, dtype=float) >= threshold).astype(int)
    profit = expected_profit(y_true_test, y_pred_test, cost_fp, cost_fn)
    tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1]).ravel()
    return {
        "model": name,
        "status": "ok",
        "threshold": threshold,
        "expected_profit": profit,
        "accuracy": accuracy_score(y_true_test, y_pred_test),
        "precision": precision_score(y_true_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_true_test, y_pred_test, zero_division=0),
        "f_beta": custom_scorer(y_true_test, y_pred_test, cost_fp=cost_fp, cost_fn=cost_fn),
        "roc_auc": _safe_auc(y_true_test, y_prob_test),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "calls": int(y_pred_test.sum()),
    }
 
 

# Evaluation
def compare_against_baselines(
    y_true: pd.Series | np.ndarray,
    cost_fp: float,
    cost_fn: float,
    logistic_row: dict[str, Any] | None,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    return {
        "call_everyone_profit": expected_profit(y_true_arr, np.ones_like(y_true_arr), cost_fp, cost_fn),
        "logistic_profit": float(logistic_row["expected_profit"]) if logistic_row else float("nan"),
    }
 
 
def call_everyone_baseline_row(
    y_true: pd.Series | np.ndarray,
    cost_fp: float,
    cost_fn: float,
    logistic_profit: float = float("nan"),
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred = np.ones_like(y_true_arr)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    profit = expected_profit(y_true_arr, y_pred, cost_fp, cost_fn)
    return {
        "model": "Call Everyone",
        "status": "baseline",
        "threshold": np.nan,
        "expected_profit": profit,
        "accuracy": accuracy_score(y_true_arr, y_pred),
        "precision": precision_score(y_true_arr, y_pred, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred, zero_division=0),
        "f_beta": custom_scorer(y_true_arr, y_pred, cost_fp=cost_fp, cost_fn=cost_fn),
        "roc_auc": np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "calls": int(y_pred.sum()),
        "profit_gain_vs_logistic": profit - logistic_profit,
        "profit_gain_vs_call_everyone": 0.0,
        "call_everyone_profit": profit,
        "logistic_profit": logistic_profit,
    }
 
 
def evaluate_at_profits(
    bundle: TrainingBundle,
    cost_fp: float,
    cost_fn: float,
) -> pd.DataFrame:
    rows = []
    for name, prob_test in bundle.probabilities.items():
        prob_train = bundle.train_probabilities.get(name)
        if prob_train is None:
            # Fallback: if train probabilities are missing for this model,
            # skip it rather than silently leaking the test set.
            continue
        rows.append(_metrics_row(
            name,
            y_true_train=bundle.y_train,
            y_prob_train=prob_train,
            y_true_test=bundle.y_test,
            y_prob_test=prob_test,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        ))
 
    for name, run in bundle.models.items():
        if run.status != "ok":
            rows.append({
                "model": name,
                "status": run.status,
                "error": run.error,
                "expected_profit": np.nan,
            })
 
    if not rows:
        return pd.DataFrame()
 
    results = pd.DataFrame(rows)
    valid = results[results["status"].eq("ok")].copy()
    logistic_row = None
    if not valid.empty and "Logistic Regression" in valid["model"].values:
        logistic_row = valid.loc[valid["model"].eq("Logistic Regression")].iloc[0].to_dict()
 
    baselines = compare_against_baselines(bundle.y_test, cost_fp, cost_fn, logistic_row)
    # profit_gain_vs_* is positive when this model earns more than the baseline.
    results["profit_gain_vs_logistic"] = results["expected_profit"] - baselines["logistic_profit"]
    results["profit_gain_vs_call_everyone"] = results["expected_profit"] - baselines["call_everyone_profit"]
    results["call_everyone_profit"] = baselines["call_everyone_profit"]
    results["logistic_profit"] = baselines["logistic_profit"]
    # Sort: ok rows first (status desc), then by expected_profit descending (higher is better).
    return results.sort_values(
        ["status", "expected_profit"], ascending=[False, False], na_position="last"
    )
 
 
def focused_comparison_rows(
    results: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> pd.DataFrame:
    valid = results[results["status"].eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame()
 
    selected = []
    best = valid.sort_values("expected_profit", ascending=False).iloc[[0]]
    selected.append(best)
 
    second_best = valid[~valid["model"].eq(best.iloc[0]["model"])].head(1)
    if not second_best.empty:
        selected.append(second_best)
 
    logistic = valid[valid["model"].eq("Logistic Regression")]
    if not logistic.empty and logistic.iloc[0]["model"] not in pd.concat(selected)["model"].values:
        selected.append(logistic.iloc[[0]])
 
    focused = pd.concat(selected, ignore_index=True)
    logistic_profit = float(logistic.iloc[0]["expected_profit"]) if not logistic.empty else float("nan")
    call_everyone = pd.DataFrame([call_everyone_baseline_row(y_true, cost_fp, cost_fn, logistic_profit)])
    focused = pd.concat([focused, call_everyone], ignore_index=True)
    focused = focused.drop_duplicates(subset=["model"], keep="first")
    # profit_gain_vs_best: positive means this row earns more than the best model.
    focused["profit_gain_vs_best"] = focused["expected_profit"] - float(best.iloc[0]["expected_profit"])
    return focused
 
 
def recommended_model_row(results: pd.DataFrame) -> dict[str, Any] | None:
    valid = results[results["status"].eq("ok")].copy()
    if valid.empty:
        return None
    return valid.sort_values("expected_profit", ascending=False).iloc[0].to_dict()
 
 

# Sensitivity analysis
def cost_sensitivity(
    bundle: TrainingBundle,
    cost_fp: float = 1.0,
    ratios: np.ndarray | None = None,
) -> pd.DataFrame:
    if ratios is None:
        ratios = np.round(np.linspace(10, 30, 21), 2)
    rows = []
    for ratio in ratios:
        result = evaluate_at_profits(bundle, cost_fp=cost_fp, cost_fn=cost_fp * float(ratio))
        valid = result[result["status"].eq("ok")]
        if valid.empty:
            continue
        # evaluate_at_profits returns ok rows sorted by expected_profit descending.
        best = valid.iloc[0]
        rows.append({
            "cost_ratio": float(ratio),
            "best_model": best["model"],
            "best_profit": float(best["expected_profit"]),
            "logistic_profit": float(best["logistic_profit"]),
            "call_everyone_profit": float(best["call_everyone_profit"]),
            "ml_profit_gain_vs_call_everyone": float(best["profit_gain_vs_call_everyone"]),
            "ml_profit_gain_vs_logistic": float(best["profit_gain_vs_logistic"]),
        })
    return pd.DataFrame(rows)
 
 
def cost_sensitivity_by_model(
    bundle: TrainingBundle,
    cost_fp: float = 1.0,
    ratios: np.ndarray | None = None,
) -> pd.DataFrame:
    if ratios is None:
        ratios = np.round(np.linspace(10, 30, 21), 2)
    rows = []
    for ratio in ratios:
        result = evaluate_at_profits(bundle, cost_fp=cost_fp, cost_fn=cost_fp * float(ratio))
        valid = result[result["status"].eq("ok")].copy()
        if valid.empty:
            continue
        # Rank 1 = highest profit.
        valid["rank"] = valid["expected_profit"].rank(method="dense", ascending=False).astype(int)
        for _, row in valid.iterrows():
            rows.append({
                "cost_ratio": float(ratio),
                "model": row["model"],
                "rank": int(row["rank"]),
                "expected_profit": float(row["expected_profit"]),
                "threshold": float(row["threshold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "profit_gain_vs_call_everyone": float(row["profit_gain_vs_call_everyone"]),
                "profit_gain_vs_logistic": float(row["profit_gain_vs_logistic"]),
            })
    return pd.DataFrame(rows)
 
 

# Manual / external input helpers
def _default_raw_values(reference: pd.DataFrame | None = None) -> dict[str, Any]:
    source = reference if reference is not None else prep.load_data()
    defaults: dict[str, Any] = {}
    for col in _ALL_INPUT_FEATURES:
        if col not in source.columns:
            defaults[col] = "unknown"
            continue
        series = source[col]
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            defaults[col] = float(clean_series.median()) if not clean_series.empty else 0.0
        else:
            mode = series.dropna().astype(str).mode()
            defaults[col] = mode.iloc[0] if not mode.empty else "unknown"
    return defaults
 
 
def _complete_raw_input_frame(
    input_df: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    row = input_df.copy()
    if TARGET_COLUMN in row.columns:
        row = row.drop(columns=[TARGET_COLUMN])
    defaults = _default_raw_values(reference)
    for col in _ALL_INPUT_FEATURES:
        if col not in row.columns:
            row[col] = defaults[col]
        elif row[col].isna().any():
            row[col] = row[col].fillna(defaults[col])
    return row[_ALL_INPUT_FEATURES].copy()
 
 
def input_field_hints() -> dict[str, str]:
    return {
        "age": "Age in years",
        "job": "Job type (e.g., admin., technician)",
        "marital": "Marital status (married/single/divorced)",
        "education": "Education level (basic.4y, high.school, university.degree, etc.)",
        "default": "Has credit in default? (yes/no)",
        "housing": "Has housing loan? (yes/no)",
        "loan": "Has personal loan? (yes/no)",
        "contact": "Contact communication type (telephone, cellular)",
        "month": "Last contact month (jan, feb, ...)",
        "day_of_week": "Last contact day of week (mon, tue, ...)",
        "duration": "Last contact duration in seconds (dropped during modelling)",
        "campaign": "Number of contacts performed during this campaign for this client",
        "pdays": "Days passed after last contact (999 means not previously contacted)",
        "previous": "Number of contacts performed before this campaign",
        "emp.var.rate": "Employment variation rate (macro)",
        "cons.price.idx": "Consumer price index",
        "cons.conf.idx": "Consumer confidence index",
        "euribor3m": "3-month Euribor rate",
        "nr.employed": "Number of employees (macro)",
    }
 
 

# SHAP explainability
def shap_explanation(
    bundle: TrainingBundle,
    model_name: str,
    input_df: pd.DataFrame | None = None,
    sample_size: int = 80,
) -> tuple[Any, pd.DataFrame]:
    run = bundle.models.get(model_name)
    if run is None or run.estimator is None:
        raise ValueError(f"Model is not available: {model_name}")
 
    if input_df is None:
        X_raw = bundle.raw_test_frame.sample(
            min(sample_size, len(bundle.raw_test_frame)), random_state=RANDOM_STATE
        )
    else:
        X_raw = _complete_raw_input_frame(input_df)
 
    return ml_module.get_shap_explanation_for_pipeline(
        run.estimator, X_raw, sample_size=sample_size
    )
 
 
def probability_shap_explanation(
    bundle: TrainingBundle,
    model_name: str,
    input_df: pd.DataFrame,
    background_size: int = 50,
) -> tuple[Any, pd.DataFrame, float]:
    run = bundle.models.get(model_name)
    if run is None or run.estimator is None:
        raise ValueError(f"Model is not available: {model_name}")
 
    X_raw = _complete_raw_input_frame(input_df)
    return ml_module.get_probability_shap_explanation_for_pipeline(
        run.estimator, X_raw, background_size=background_size, background_raw=bundle.raw_test_frame
    )
 
 
def feature_importance(
    bundle: TrainingBundle,
    model_name: str,
    max_features: int = 12,
) -> tuple[pd.DataFrame, str]:
    run = bundle.models.get(model_name)
    if run is None or run.estimator is None:
        return pd.DataFrame(columns=["feature", "importance"]), "unavailable"
 
    # --- SHAP importance ---
    try:
        import shap
 
        sample = bundle.raw_test_frame.sample(
            min(120, len(bundle.raw_test_frame)), random_state=RANDOM_STATE
        )
        names = run.estimator.named_steps["preprocess"].get_feature_names_out()
        transformed_arr = run.estimator.named_steps["preprocess"].transform(
            prep.clean_raw_data(sample)
        )
        transformed = pd.DataFrame(transformed_arr, columns=names, index=sample.index)
        model = run.estimator.named_steps["model"]
        explainer = shap.Explainer(model, transformed)
        values = explainer(transformed)
        raw_values = values.values
        if raw_values.ndim == 3:
            raw_values = raw_values[:, :, -1]
        importance = np.abs(raw_values).mean(axis=0)
        out = pd.DataFrame({"feature": names, "importance": importance})
        return out.sort_values("importance", ascending=False).head(max_features), "shap"
    except Exception:
        pass
 
    # --- Permutation importance fallback ---
    try:
        sample_size = min(700, len(bundle.raw_test_frame))
        sample_X_raw = bundle.raw_test_frame.sample(sample_size, random_state=RANDOM_STATE)
        sample_y = bundle.y_test.loc[sample_X_raw.index]
        # Permutation importance needs the bundle to accept raw input; FittedModelBundle
        # handles cleaning internally so we pass raw rows directly.
        result = permutation_importance(
            run.estimator,
            sample_X_raw,
            sample_y,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="roc_auc",
        )
        out = pd.DataFrame({"feature": bundle.feature_names, "importance": result.importances_mean})
        return out.sort_values("importance", ascending=False).head(max_features), "permutation"
    except Exception:
        pass
 
    # --- Native model coefficients / feature_importances_ ---
    model = run.estimator.named_steps["model"]
    preprocessor = run.estimator.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array(bundle.feature_names)
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"]), "unavailable"
    out = pd.DataFrame({"feature": names[: len(importance)], "importance": importance})
    return out.sort_values("importance", ascending=False).head(max_features), "model-native"
 
 

# Customer scoring
def score_customers(
    bundle: TrainingBundle,
    model_name: str,
    raw_customers: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    run = bundle.models[model_name]
    if run.estimator is None:
        raise ValueError(f"Model is not available: {model_name}")
    X_raw = _complete_raw_input_frame(raw_customers)
    probabilities = ml_module.get_positive_probabilities(run.estimator, X_raw)
    decision = np.where(probabilities >= threshold, "Call", "Do not call")
    confidence = np.where(
        np.abs(probabilities - threshold) >= 0.25,
        "High",
        np.where(np.abs(probabilities - threshold) >= 0.1, "Medium", "Low"),
    )
    return pd.DataFrame({
        "probability": probabilities,
        "decision": decision,
        "confidence": confidence,
        "threshold": threshold
    })
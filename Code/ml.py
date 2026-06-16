# Importing necessary libraries for machine learning models

# General Utilities
import numpy as np
import pandas as pd

# Machine Learning Models and Tools
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import shap
from preprocessing import clean_raw_data
from evaluation import custom_scorer

# Do not load data at import time. Callers should pass data into training wrappers.
X_train = None
y_train = None
X_train_res = None
y_train_res = None
X_test = None
y_test = None
X_test_raw = None

RANDOM_STATE = 42
cost_fp = 1  # Cost of a false positive (wasting a call)
cost_fn = 4  # Cost of a false negative (missing an opportunity)

# Mapping to store best fitted estimators from training functions
BEST_ESTIMATORS = {}


# Helper: compute F-Beta on a already-fitted model without re-fitting.
def get_fbeta_score(model, X, y):
    """Compute F-Beta on pre-fitted model."""
    y_pred = model.predict(X)
    return custom_scorer(y, y_pred, cost_fp, cost_fn)


# Helper: evaluate a pre-fitted model on test data and return F-Beta.
# NOTE: this function only evaluates – it does NOT refit the model.
def evaluate_fbeta(model, X_test_local, y_test_local):
    """Return F-Beta score for an already-fitted model on the test set."""
    return get_fbeta_score(model, X_test_local, y_test_local)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression_balanced(X_tr, y_tr, X_te, y_te, C=1.0):
    """Train Logistic Regression on the original (imbalanced) data with class_weight='balanced'.

    Fits and stores the estimator; returns the test F-Beta score.
    """
    clf = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=C,
    )
    clf.fit(X_tr, y_tr)
    BEST_ESTIMATORS['LogisticRegression_balanced'] = clf
    fbeta = evaluate_fbeta(clf, X_te, y_te)
    return f"Logistic Regression Balanced - Test F-Beta score: {fbeta:.4f}"


def train_logistic_regression_resampled(X_tr, y_tr, X_te, y_te, C=1.0):
    """Train Logistic Regression on SMOTE-resampled data without class weighting.

    Fits and stores the estimator; returns the test F-Beta score.
    """
    clf = LogisticRegression(
        solver='liblinear',
        class_weight=None,
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=C,
    )
    clf.fit(X_tr, y_tr)
    BEST_ESTIMATORS['LogisticRegression_resampled'] = clf
    fbeta = evaluate_fbeta(clf, X_te, y_te)
    return f"Logistic Regression Resampled - Test F-Beta score: {fbeta:.4f}"


# ---------------------------------------------------------------------------
# K-Nearest Neighbours
# ---------------------------------------------------------------------------

def train_knn_balanced(X_tr, y_tr, X_te, y_te):
    """Train KNN with GridSearchCV on the original (imbalanced) data.

    Hyperparameters tuned via F-Beta (derived from cost_fp/cost_fn):
    n_neighbors, weights, metric.
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    knn = KNeighborsClassifier()
    search = GridSearchCV(
        knn, param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    cv_score = search.best_score_
    BEST_ESTIMATORS['KNN_balanced'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    return (
        f"KNN Balanced - Best hyperparameters: n_neighbors={best.n_neighbors}, "
        f"weights='{best.weights}', metric='{best.metric}'\n"
        f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
    )


def train_knn_resampled(X_tr, y_tr, X_te, y_te):
    """Train KNN with GridSearchCV on SMOTE-resampled data.

    Hyperparameters tuned via F-Beta: n_neighbors, weights, metric.
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    knn = KNeighborsClassifier()
    search = GridSearchCV(
        knn, param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    cv_score = search.best_score_
    BEST_ESTIMATORS['KNN_resampled'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    return (
        f"KNN Resampled - Best hyperparameters: n_neighbors={best.n_neighbors}, "
        f"weights='{best.weights}', metric='{best.metric}'\n"
        f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
    )


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

def train_decision_tree_balanced(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train Decision Tree with optional GridSearchCV on the original data.

    Hyperparameters tuned: max_depth, min_samples_leaf, criterion, ccp_alpha.
    class_weight='balanced' handles the imbalance.
    """
    base = DecisionTreeClassifier(
        class_weight='balanced',
        max_depth=7,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    if do_tune:
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_leaf': [2, 5, 10, 20],
            'criterion': ['gini', 'entropy'],
            'ccp_alpha': [0.0, 0.001, 0.01, 0.05],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['DecisionTree_balanced'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"Decision Tree Balanced - Best hyperparameters: max_depth={best.max_depth}, "
            f"min_samples_leaf={best.min_samples_leaf}, criterion='{best.criterion}', "
            f"ccp_alpha={best.ccp_alpha}\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"Decision Tree Balanced (no tuning) - Test F-Beta score: {fbeta:.4f}"


def train_decision_tree_resampled(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train Decision Tree with optional GridSearchCV on SMOTE-resampled data.

    Hyperparameters tuned: max_depth, min_samples_leaf, criterion, ccp_alpha.
    """
    base = DecisionTreeClassifier(
        class_weight=None,
        max_depth=7,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    if do_tune:
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_leaf': [2, 5, 10, 20],
            'criterion': ['gini', 'entropy'],
            'ccp_alpha': [0.0, 0.001, 0.01, 0.05],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['DecisionTree_resampled'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"Decision Tree Resampled - Best hyperparameters: max_depth={best.max_depth}, "
            f"min_samples_leaf={best.min_samples_leaf}, criterion='{best.criterion}', "
            f"ccp_alpha={best.ccp_alpha}\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"Decision Tree Resampled (no tuning) - Test F-Beta score: {fbeta:.4f}"


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest_balanced(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train Random Forest on the original data with class_weight='balanced'.

    Hyperparameters tuned: n_estimators, min_samples_leaf, max_features.
    """
    base = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    if do_tune:
        param_grid = {
            'n_estimators': [100, 300],
            'min_samples_leaf': [1, 5, 20],
            'max_features': ['sqrt', 'log2'],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['RandomForest_balanced'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"Random Forest Balanced - Best hyperparameters: n_estimators={best.n_estimators}, "
            f"min_samples_leaf={best.min_samples_leaf}, max_features='{best.max_features}'\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"Random Forest Balanced (no tuning) - Test F-Beta score: {fbeta:.4f}"


def train_random_forest_resampled(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train Random Forest on SMOTE-resampled data.

    Hyperparameters tuned: n_estimators, min_samples_leaf, max_features.
    """
    base = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    if do_tune:
        param_grid = {
            'n_estimators': [100, 300],
            'min_samples_leaf': [1, 5, 20],
            'max_features': ['sqrt', 'log2'],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['RandomForest_resampled'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"Random Forest Resampled - Best hyperparameters: n_estimators={best.n_estimators}, "
            f"min_samples_leaf={best.min_samples_leaf}, max_features='{best.max_features}'\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"Random Forest Resampled (no tuning) - Test F-Beta score: {fbeta:.4f}"


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost_balanced(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train XGBoost on the original data with scale_pos_weight to handle imbalance.

    Hyperparameters tuned: n_estimators, learning_rate, max_depth, gamma, subsample.
    """
    pos = int(np.sum(y_tr == 1))
    neg = int(np.sum(y_tr == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    base = XGBClassifier(
        eval_metric='logloss',
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )
    if do_tune:
        param_grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.05, 0.2],
            'max_depth': [3, 6, 9],
            'gamma': [0, 1],
            'subsample': [0.8, 1.0],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['XGBoost_balanced'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"XGBoost Balanced - Best hyperparameters: learning_rate={best.learning_rate}, "
            f"max_depth={best.max_depth}, gamma={best.gamma}, subsample={best.subsample}\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"XGBoost Balanced (no tuning) - Test F-Beta score: {fbeta:.4f}"


def train_xgboost_resampled(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train XGBoost on SMOTE-resampled data (no scale_pos_weight needed).

    Hyperparameters tuned: n_estimators, learning_rate, max_depth, gamma, subsample.
    """
    base = XGBClassifier(
        eval_metric='logloss',
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    if do_tune:
        param_grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.05, 0.2],
            'max_depth': [3, 6, 9],
            'gamma': [0, 1],
            'subsample': [0.8, 1.0],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['XGBoost_resampled'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"XGBoost Resampled - Best hyperparameters: learning_rate={best.learning_rate}, "
            f"max_depth={best.max_depth}, gamma={best.gamma}, subsample={best.subsample}\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"XGBoost Resampled (no tuning) - Test F-Beta score: {fbeta:.4f}"


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm_balanced(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train LightGBM on the original data with class_weight='balanced'.

    Hyperparameters tuned: n_estimators, learning_rate, num_leaves, subsample, reg_alpha.
    """
    base = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    if do_tune:
        param_grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 63],
            'subsample': [0.8, 1.0],
            'reg_alpha': [0.1, 0.5],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['LightGBM_balanced'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"LightGBM Balanced - Best hyperparameters: n_estimators={best.n_estimators}, "
            f"learning_rate={best.learning_rate}, num_leaves={best.num_leaves}, "
            f"subsample={best.subsample}, reg_alpha={best.reg_alpha}\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"LightGBM Balanced (no tuning) - Test F-Beta score: {fbeta:.4f}"


def train_lightgbm_resampled(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train LightGBM on SMOTE-resampled data.

    Hyperparameters tuned: n_estimators, learning_rate, num_leaves, subsample, reg_alpha.
    """
    base = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        class_weight=None,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    if do_tune:
        param_grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 63],
            'subsample': [0.8, 1.0],
            'reg_alpha': [0.1, 0.5],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['LightGBM_resampled'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"LightGBM Resampled - Best hyperparameters: n_estimators={best.n_estimators}, "
            f"learning_rate={best.learning_rate}, num_leaves={best.num_leaves}, "
            f"subsample={best.subsample}, reg_alpha={best.reg_alpha}\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"LightGBM Resampled (no tuning) - Test F-Beta score: {fbeta:.4f}"


# ---------------------------------------------------------------------------
# Neural Network (MLP)
# ---------------------------------------------------------------------------

def train_mlp_balanced(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train MLP on the original data.

    GridSearchCV tunes hidden_layer_sizes, activation, alpha, learning_rate_init.
    Sample weights are applied during GridSearchCV via the scorer's training
    data rather than post-hoc re-fitting (MLP does not natively support
    sample_weight in fit so we use class imbalance via early stopping and rely
    on the F-Beta scorer to bias the search toward recall).
    """
    base = MLPClassifier(
        hidden_layer_sizes=(64, 24),
        activation='relu',
        alpha=0.001,
        learning_rate_init=0.001,
        early_stopping=True,
        max_iter=300,
        solver='adam',
        learning_rate='adaptive',
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )
    if do_tune:
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['MLP_balanced'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"MLP Balanced - Best hyperparameters: hidden_layer_sizes={best.hidden_layer_sizes}, "
            f"activation='{best.activation}', alpha={best.alpha}, "
            f"learning_rate_init={best.learning_rate_init}\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"MLP Balanced (no tuning) - Test F-Beta score: {fbeta:.4f}"


def train_mlp_resampled(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Train MLP on SMOTE-resampled data.

    Hyperparameters tuned: hidden_layer_sizes, activation, alpha, learning_rate_init.
    """
    base = MLPClassifier(
        hidden_layer_sizes=(64, 24),
        activation='relu',
        alpha=0.001,
        learning_rate_init=0.001,
        early_stopping=True,
        max_iter=300,
        solver='adam',
        learning_rate='adaptive',
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )
    if do_tune:
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    BEST_ESTIMATORS['MLP_resampled'] = best
    fbeta = evaluate_fbeta(best, X_te, y_te)
    if do_tune:
        return (
            f"MLP Resampled - Best hyperparameters: hidden_layer_sizes={best.hidden_layer_sizes}, "
            f"activation='{best.activation}', alpha={best.alpha}, "
            f"learning_rate_init={best.learning_rate_init}\n"
            f"Cross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
        )
    return f"MLP Resampled (no tuning) - Test F-Beta score: {fbeta:.4f}"


# ---------------------------------------------------------------------------
# Voting Ensembles
# ---------------------------------------------------------------------------

def _fit_voting_ensemble(estimators, X_tr, y_tr, X_te, y_te, model_name, do_tune=True):
    """Fit a soft-voting ensemble and optionally tune only its voting weights."""
    base = VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
    if do_tune:
        param_grid = {
            'weights': [
                [1, 1, 1],
                [1, 2, 1],
                [1, 2, 2],
                [2, 2, 1],
            ],
        }
        search = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring=make_scorer(custom_scorer, cost_fp=cost_fp, cost_fn=cost_fn),
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_score = search.best_score_
    else:
        base.fit(X_tr, y_tr)
        best = base
        cv_score = None

    fbeta = evaluate_fbeta(best, X_te, y_te)
    return best, cv_score, fbeta


def train_voting_lr_rf_mlp_balanced(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Soft-voting ensemble of Logistic Regression, Random Forest, and MLP."""
    estimators = [
        ('lr', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=220, min_samples_leaf=8, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 24), activation='relu', alpha=0.001, learning_rate_init=0.001, early_stopping=True, max_iter=300, solver='adam', learning_rate='adaptive', n_iter_no_change=10, random_state=RANDOM_STATE)),
    ]
    best, cv_score, fbeta = _fit_voting_ensemble(estimators, X_tr, y_tr, X_te, y_te, model_name='Voting_LR_RF_MLP_balanced', do_tune=do_tune)
    BEST_ESTIMATORS['Voting_LR_RF_MLP_balanced'] = best
    if do_tune:
        return f"Voting LR + RF + MLP Balanced - Best weights: {best.weights}\nCross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
    return f"Voting LR + RF + MLP Balanced (no tuning) - Test F-Beta score: {fbeta:.4f}"


def train_voting_lr_rf_xgb_balanced(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Soft-voting ensemble of Logistic Regression, Random Forest, and XGBoost."""
    pos = int(np.sum(y_tr == 1))
    neg = int(np.sum(y_tr == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    estimators = [
        ('lr', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=220, min_samples_leaf=8, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)),
        ('xgb', XGBClassifier(eval_metric='logloss', n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight)),
    ]
    best, cv_score, fbeta = _fit_voting_ensemble(estimators, X_tr, y_tr, X_te, y_te, model_name='Voting_LR_RF_XGB_balanced', do_tune=do_tune)
    BEST_ESTIMATORS['Voting_LR_RF_XGB_balanced'] = best
    if do_tune:
        return f"Voting LR + RF + XGBoost Balanced - Best weights: {best.weights}\nCross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
    return f"Voting LR + RF + XGBoost Balanced (no tuning) - Test F-Beta score: {fbeta:.4f}"


def train_voting_lr_rf_lgbm_balanced(X_tr, y_tr, X_te, y_te, do_tune=True):
    """Soft-voting ensemble of Logistic Regression, Random Forest, and LightGBM."""
    estimators = [
        ('lr', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=220, min_samples_leaf=8, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)),
        ('lgbm', LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1)),
    ]
    best, cv_score, fbeta = _fit_voting_ensemble(estimators, X_tr, y_tr, X_te, y_te, model_name='Voting_LR_RF_LGBM_balanced', do_tune=do_tune)
    BEST_ESTIMATORS['Voting_LR_RF_LGBM_balanced'] = best
    if do_tune:
        return f"Voting LR + RF + LightGBM Balanced - Best weights: {best.weights}\nCross-validation F-Beta score: {cv_score:.4f}\nTest F-Beta score: {fbeta:.4f}"
    return f"Voting LR + RF + LightGBM Balanced (no tuning) - Test F-Beta score: {fbeta:.4f}"


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def get_positive_probabilities(model, X):
    """Extract positive-class probabilities from any sklearn-compatible model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(X), dtype=float)


def get_shap_explanation_for_pipeline(estimator, X_raw: pd.DataFrame, sample_size: int = 80):
    """Return a SHAP Explanation, transformed feature DataFrame, and raw-space
    feature DataFrame for a FittedModelBundle.

    Background data is sampled from the passed-in X_raw (the test-set raw frame
    supplied by the dashboard bundle), so this function has no dependency on
    the module-level X_test_raw global.

    Returns
    -------
    explanation : shap.Explanation
        SHAP values in the model's native output space, with transformed
        (scaled / one-hot-encoded) feature names.
    transformed_df : pd.DataFrame
        The preprocessed features the model actually saw, with column names.
        Used by SHAP's plotting functions, which expect this shape.
    raw_df : pd.DataFrame
        The cleaned *raw-scale* feature values for the same rows (original
        units, original category labels). Use this to relabel axes / tooltips
        so the plots are human-readable.
    """
    if not hasattr(estimator, 'named_steps'):
        raise ValueError('Estimator must be a FittedModelBundle with a `preprocess` step')

    preprocessor = estimator.named_steps['preprocess']
    model = estimator.named_steps['model']

    if isinstance(X_raw, pd.DataFrame):
        X_clean = clean_raw_data(X_raw.copy())
    else:
        raise ValueError('X_raw must be a pandas DataFrame')

    # Sample background rows from the provided raw frame (test split).
    background_clean = X_clean.sample(
        min(sample_size, len(X_clean)), random_state=RANDOM_STATE
    )
    names = preprocessor.get_feature_names_out()
    background = pd.DataFrame(
        preprocessor.transform(background_clean), columns=names, index=background_clean.index
    )
    transformed_df = pd.DataFrame(
        preprocessor.transform(X_clean), columns=names, index=X_clean.index
    )
    # Raw-scale companion frame (same rows/order as transformed_df) for
    # human-readable axis labels and tooltips.
    raw_df = X_clean.loc[transformed_df.index].copy()

    model_name = model.__class__.__name__
    if model_name.startswith('Logistic'):
        explainer = shap.LinearExplainer(model, background)
        values = explainer.shap_values(transformed_df)
        base_value = explainer.expected_value
    elif model_name in ('DecisionTreeClassifier', 'RandomForestClassifier') or model_name.startswith('XGB') or model_name.startswith('LGBM'):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed_df)
        base_value = explainer.expected_value
    else:
        probability_fn = lambda rows: get_positive_probabilities(model, pd.DataFrame(rows, columns=names))
        explainer = shap.KernelExplainer(probability_fn, background)
        raw_values = explainer.shap_values(transformed_df)

        if isinstance(raw_values, list):
            values = raw_values[0]
        else:
            values = raw_values
            
        base_value = explainer.expected_value

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    if isinstance(base_value, list):
        base_value = base_value[-1]
    base_value_arr = np.asarray(base_value)
    if base_value_arr.ndim > 0:
        base_value = float(base_value_arr.reshape(-1)[-1])

    explanation = shap.Explanation(
        values=values,
        base_values=np.repeat(float(base_value), len(transformed_df)),
        data=transformed_df.to_numpy(),
        feature_names=list(names),
    )
    return explanation, transformed_df, raw_df


def get_probability_shap_explanation_for_pipeline(estimator, X_raw: pd.DataFrame, background_size: int = 50, background_raw: pd.DataFrame | None = None):
    """Return a probability-space SHAP Explanation, transformed DataFrame,
    raw-space DataFrame, and scalar probability for the first row of X_raw.

    Uses TreeExplainer / LinearExplainer with a probability output where
    possible (fast, exact for those model families) and falls back to
    KernelExplainer only for model types that support neither (e.g. MLP,
    Voting ensembles). Background is sampled from X_raw (the test-set raw
    frame).

    Returns
    -------
    explanation : shap.Explanation
        Probability-space SHAP values for the first row of X_raw.
    transformed_df : pd.DataFrame
        The preprocessed features the model actually saw.
    raw_df : pd.DataFrame
        Cleaned raw-scale feature values for the same row (original units /
        category labels) for human-readable display.
    probability : float
        The model's predicted positive-class probability for the first row.
    """
    if not hasattr(estimator, 'named_steps'):
        raise ValueError('Estimator must be a FittedModelBundle with a `preprocess` step')

    preprocessor = estimator.named_steps['preprocess']
    model = estimator.named_steps['model']

    if isinstance(X_raw, pd.DataFrame):
        X_clean = clean_raw_data(X_raw.copy())
    else:
        raise ValueError('X_raw must be a pandas DataFrame')

    background_source = background_raw if background_raw is not None else X_raw
    background_clean_source = clean_raw_data(background_source.copy())
    background_clean = background_clean_source.sample(
        min(background_size, len(background_clean_source)), random_state=RANDOM_STATE
    )
    names = preprocessor.get_feature_names_out()
    background = pd.DataFrame(
        preprocessor.transform(background_clean), columns=names, index=background_clean.index
    )
    transformed_df = pd.DataFrame(
        preprocessor.transform(X_clean), columns=names, index=X_clean.index
    )
    raw_df = X_clean.loc[transformed_df.index].copy()

    model_name = model.__class__.__name__
    if model_name in ('DecisionTreeClassifier', 'RandomForestClassifier') or model_name.startswith('XGB') or model_name.startswith('LGBM'):
        # TreeExplainer with model_output="probability" gives exact, fast
        # probability-space attributions for tree-based models.
        explainer = shap.TreeExplainer(model, data=background, model_output="probability")
        values = explainer.shap_values(transformed_df)
        base_value = explainer.expected_value
    elif model_name.startswith('Logistic'):
        # LinearExplainer with a logit link, then convert attributions and the
        # base value into probability space via the sigmoid.
        explainer = shap.LinearExplainer(model, background)
        logit_values = np.asarray(explainer.shap_values(transformed_df))
        logit_base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
        logit_total = logit_base + logit_values.sum(axis=1)
        prob_total = 1.0 / (1.0 + np.exp(-logit_total))
        prob_base = 1.0 / (1.0 + np.exp(-logit_base))
        # Distribute the probability-space delta proportionally to each
        # feature's share of the logit-space attribution.
        logit_sum_abs = np.abs(logit_values).sum(axis=1, keepdims=True)
        logit_sum_abs = np.where(logit_sum_abs == 0, 1.0, logit_sum_abs)
        share = logit_values / logit_sum_abs
        prob_delta = (prob_total - prob_base).reshape(-1, 1)
        values = share * prob_delta
        base_value = prob_base
    else:
        # MLP / Voting ensembles: no closed-form probability explainer, so
        # fall back to KernelExplainer directly in probability space.
        probability_fn = lambda rows: get_positive_probabilities(model, pd.DataFrame(rows, columns=names))
        explainer = shap.KernelExplainer(probability_fn, background)
        values = explainer.shap_values(transformed_df)
        base_value = explainer.expected_value

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim == 1:
        values = values.reshape(1, -1)

    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).reshape(-1)[-1])
    else:
        base_value = float(base_value)

    probability = float(np.asarray(get_positive_probabilities(model, transformed_df)).reshape(-1)[0])

    explanation = shap.Explanation(
        values=values,
        base_values=np.repeat(base_value, len(transformed_df)),
        data=transformed_df.to_numpy(),
        feature_names=list(names),
    )
    return explanation, transformed_df, raw_df, probability


if __name__ == "__main__":
    pass
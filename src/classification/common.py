"""Shared pieces for the classification task (IsCanceled).

Everything the other classification files need lives here:
paths, data loading, the model list, the pipeline, and the metrics.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from xgboost import XGBClassifier

# -------------------settings-------------------------------------------------------------
SEED = 42
TARGET = "IsCanceled"
DEPLOYED_MODEL = "random_forest"      #the model used by default unless --model is given

ROOT = Path(__file__).resolve().parents[2]          # project root (folder with main.py)
DATA_DIR = ROOT / "src" / "data" / "classificationdata" #Folder containing the datasets
TRAIN_PATH = DATA_DIR / "H2_classification_train_with_agent.csv" #Training CSV
TEST_PATH = DATA_DIR / "H2_classification_test_with_agent.csv" #Testing CSV
ARTIFACTS = ROOT / "artifacts"#saved models
OUTPUTS = ROOT / "outputs"  #Where result files are saved

#same 5-fold split everywhere, so every comparison is fair
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
#These are the possible classification thresholds you will test later
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2)


# ---------------------models---------------------------------------------------------
# For each model: how to build it, whether it needs scaled numbers,
# its default (best tuned) settings, and the search space used by `tune`.
MODELS = {
    "random_forest": {
        "label": "RandomForest",
        "scale": False,
        "make": lambda **p: RandomForestClassifier(random_state=SEED, n_jobs=-1, **p),
        #Initial/best settings to use if no tuning file exists
        "default_params": {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
        #Hyperparameter values to explore
        "space": {"n_estimators": [100, 200, 300, 500],
                  "max_depth": [10, 20, 30, None],
                  "min_samples_leaf": [1, 2, 5, 10]},
        "n_iter": 20, #Number of random combinations to test
    },
    "xgboost": {
        "label": "XGBoost",
        "scale": False,
        "make": lambda **p: XGBClassifier(random_state=SEED, eval_metric="logloss", **p),
        "default_params": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1},
        "space": {"n_estimators": [200, 400, 600],
                  "max_depth": [3, 4, 6, 8],
                  "learning_rate": [0.03, 0.05, 0.1, 0.2]},
        "n_iter": 20,
    },
    "mlp": {
        "label": "MLP",
        #Because MLP is a neural network that is sensitive to feature scales
        "scale": True,
        "make": lambda **p: MLPClassifier(max_iter=300, early_stopping=True,
                                          random_state=SEED, **p),
        "default_params": {"hidden_layer_sizes": (100, 50), "alpha": 0.1,
                           "learning_rate_init": 0.003},
        "space": {"hidden_layer_sizes": [(50,), (100,), (100, 50), (64, 32)],
                  "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                  "learning_rate_init": [1e-3, 3e-3, 1e-2]},
        "n_iter": 10,
    },
}


# ------------------------------data--------------------------------------------------
def load_data():
    """Read the train and test files. Returns X_train, y_train, X_test, y_test."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    #Agent_te was made by the preprocessing team; we drop it and encode Agent
    #inside the pipeline instead (so the encoding never sees the test labels)
    X_train = train.drop(columns=[TARGET, "Agent_te"])
    X_test = test.drop(columns=[TARGET, "Agent_te"])
    X_train["Agent"] = X_train["Agent"].astype(str)    
    X_test["Agent"] = X_test["Agent"].astype(str)
    return X_train, train[TARGET], X_test, test[TARGET]


# -----------------------------pipeline-------------------------------------------------
def make_pipe(estimator, scale=False):
    """encoder (Agent) -> optional scaler -> model, all in one object."""
    encoder = ColumnTransformer(
        [("agent", TargetEncoder(target_type="binary",
                                smooth="auto",
                                cv=5,
                                random_state=SEED),
                                  ["Agent"])],
                                  remainder="passthrough")  #every other column passes through
    steps = [("enc", encoder)]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ----------------------------metrics---------------------------------------------------
def weighted_hm(y_true, y_pred, w_recall=2, w_spec=1): #gives Recall twice the importance of Specificity
    """Weighted harmonic mean of recall and specificity (recall counts w_recall times)."""
    recall = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    if recall == 0 or spec == 0:
        return 0.0
    return (w_recall + w_spec) / (w_recall / recall + w_spec / spec)


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision", 
    "recall": "recall", 
    "f1": "f1",
    "roc_auc": "roc_auc", 
    "pr_auc": "average_precision",
    "hm_1_1": make_scorer(weighted_hm, w_recall=1, w_spec=1),
    "hm_2_1": make_scorer(weighted_hm, w_recall=2, w_spec=1),
    "hm_3_1": make_scorer(weighted_hm, w_recall=3, w_spec=1),
}


def best_threshold(y_true, proba):
    """Cut-off that gives the best weighted harmonic mean (recall counts double)."""
    scores = [weighted_hm(y_true, (proba >= t).astype(int), 2, 1) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def metrics_row(y_true, proba, threshold):
    """All the test metrics for one model at one threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, pred),
        "hm_2_1": weighted_hm(y_true, pred, 2, 1),
        "roc_auc": roc_auc_score(y_true, proba),      # these two do not depend on the threshold
        "pr_auc": average_precision_score(y_true, proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

"""Train the final pipeline and save it (command: train)."""
import json
import time

import joblib
import sklearn
from sklearn.model_selection import cross_val_predict

from .common import (ARTIFACTS, CV5, MODELS, OUTPUTS, SEED, best_threshold,
                     load_data, make_pipe)


def load_params(model_key):
    """Use the settings found by `tune` if they exist, otherwise the defaults."""
    spec = MODELS[model_key]
    path = OUTPUTS / f"classification_best_params_{model_key}.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)["best_params"]
        if "hidden_layer_sizes" in params:           # JSON turns tuples into lists
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        return params, path.name
    return dict(spec["default_params"]), "default settings in common.py"


def train(model_key):
    spec = MODELS[model_key]
    X_train, y_train, _, _ = load_data()
    params, source = load_params(model_key)
    print(f"Training {spec['label']} with settings from: {source}")
    print("settings:", params)

    pipe = make_pipe(spec["make"](**params), spec["scale"])

    # 1) pick the threshold from out-of-fold predictions on the TRAIN data
    #    (every row is predicted by a model that never saw it)
    print("Choosing the threshold with 5-fold out-of-fold predictions...")
    oof = cross_val_predict(pipe, X_train, y_train, cv=CV5,
                            method="predict_proba", n_jobs=-1)[:, 1]
    threshold = best_threshold(y_train, oof)

    # 2) fit the final pipeline on all the train data
    start = time.time()
    pipe.fit(X_train, y_train)
    fit_seconds = time.time() - start

    # 3) save the pipeline and a small info file next to it
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(pipe, ARTIFACTS / f"classification_{model_key}.joblib")
    meta = {
        "model": model_key, "label": spec["label"], "params": params,
        "params_source": source, "threshold": threshold,
        "threshold_rule": "best weighted harmonic mean of recall and specificity "
                          "(recall counts double), on out-of-fold train predictions",
        "feature_columns": list(X_train.columns),     # the exact column order the model expects
        "known_agents": sorted(X_train["Agent"].unique().tolist()),
        "train_rows": int(len(X_train)), "seed": SEED,
        "fit_time_sec": round(fit_seconds, 2), "sklearn_version": sklearn.__version__,
    }
    with open(ARTIFACTS / f"classification_{model_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved artifacts/classification_{model_key}.joblib (threshold = {threshold})")

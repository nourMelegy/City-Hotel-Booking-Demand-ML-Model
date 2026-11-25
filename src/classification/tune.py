"""Hyperparameter search (command: tune). Uses the TRAIN data only."""
import json
import time

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from .common import CV5, MODELS, OUTPUTS, SCORING, SEED, load_data, make_pipe


def _clean(o):
    """Make numpy values safe for JSON."""
    return o.item() if hasattr(o, "item") else str(o)


def tune(model_key, n_iter=None):
    spec = MODELS[model_key]
    X_train, y_train, _, _ = load_data()  #the test file is NOT loaded into the search

    #encoder + optional scaler + model   
    pipe = make_pipe(spec["make"](), spec["scale"])

    # "model__" means: this setting belongs to the pipeline step called "model"
    space = {f"model__{k}": v 
             for k, v in spec["space"].items()}
    n_iter = n_iter or spec["n_iter"]

    search = RandomizedSearchCV(
        pipe, 
        space, 
        n_iter=n_iter, 
        cv=CV5, 
        scoring=SCORING,
        refit="pr_auc",            # the winner is the one with the best PR-AUC
        n_jobs=-1, 
        random_state=SEED, 
        verbose=1)
    start = time.time()
    search.fit(X_train, y_train)
    seconds = time.time() - start

    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"\n{spec['label']} | best CV PR-AUC: {search.best_score_:.4f} | {seconds:.0f} sec")
    print("best settings:", best)

    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(
        OUTPUTS / f"classification_tuning_{model_key}.csv", index=False)
    summary = {"model": model_key, 
               "search_space": spec["space"], 
               "n_iter": n_iter,
               "metric": "pr_auc (average_precision), 5-fold CV on train",
               "best_params": best, 
               "best_cv_pr_auc": search.best_score_,
               "tune_time_sec": seconds}
    with open(OUTPUTS / f"classification_best_params_{model_key}.json", "w") as f:
        json.dump(summary, f, indent=2, default=_clean)
    print("saved tuning table and best settings in outputs/")

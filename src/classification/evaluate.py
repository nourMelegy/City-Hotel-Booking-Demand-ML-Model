"""Score saved models on the TEST file and show saved result tables (commands: evaluate, results)."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")  #draw to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .common import ARTIFACTS, MODELS, OUTPUTS, load_data


def load_artifact(model_key):
    """Load a saved pipeline and its info file."""
    model_path = ARTIFACTS / f"classification_{model_key}.joblib"
    meta_path = ARTIFACTS / f"classification_{model_key}_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model for '{model_key}'. Run: "
            f"python main.py train --task classification --model {model_key}")
    with open(meta_path) as f:
        meta = json.load(f)
    return joblib.load(model_path), meta


def evaluate(model_key="all"):
    from .common import metrics_row
    keys = list(MODELS) if model_key == "all" else [model_key]
    _, _, X_test, y_test = load_data()

    rows = []
    for key in keys:
        try:
            pipe, meta = load_artifact(key)
        except FileNotFoundError as e:
            print("skipping", key, "->", e)
            continue
        proba = pipe.predict_proba(X_test[meta["feature_columns"]])[:, 1]
        for version, t in [("threshold 0.50", 0.50), ("saved threshold", meta["threshold"])]:
            rows.append({"model": meta["label"], "version": version, **metrics_row(y_test, proba, t)})

        # confusion matrix picture at the saved threshold
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, (proba >= meta["threshold"]).astype(int), ax=ax, colorbar=False)
        ax.set_title(f"{meta['label']} (threshold {meta['threshold']:.2f})")
        fig.tight_layout()
        OUTPUTS.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS / f"classification_confusion_{key}.png", dpi=150)
        plt.close(fig)

    if not rows:
        print("Nothing to evaluate. Train a model first.")
        return
    table = pd.DataFrame(rows).round(4)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.to_string(index=False))
    table.to_csv(OUTPUTS / "classification_evaluate_results.csv", index=False)
    print("\nsaved outputs/classification_evaluate_results.csv and the confusion matrix pictures")


def show_results():
    """Print the result tables that earlier work already saved in outputs/."""
    pd.set_option("display.width", 250, "display.max_columns", 30)
    tables = [("Baseline vs tuned (5-fold CV on train)", "classification_baseline_vs_tuned_cv.csv"),
              ("Test results", "classification_test_results.csv"),
              ("PCA / LDA comparison (CV)", "classification_pca_lda_comparison_cv.csv")]
    for title, name in tables:
        path = OUTPUTS / name
        print(f"\n=== {title} ===")
        print(pd.read_csv(path).to_string(index=False) if path.exists() else f"(missing: outputs/{name})")
    for key in MODELS:
        path = OUTPUTS / f"classification_best_params_{key}.json"
        if path.exists():
            with open(path) as f:
                s = json.load(f)
            print(f"\n=== Tuning: {key} === best CV PR-AUC {s['best_cv_pr_auc']:.4f}")
            print("search space:", s["search_space"])
            print("best settings:", s["best_params"])

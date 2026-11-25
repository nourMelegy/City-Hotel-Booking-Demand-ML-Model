"""Command line app.  Examples:
    python main.py tune     --task classification --model random_forest
    python main.py train    --task classification --model random_forest
    python main.py evaluate --task classification --model all
    python main.py predict  --task classification
    python main.py results  --task classification
"""
import argparse

MODEL_CHOICES = ["random_forest", "xgboost", "mlp"]


def run_classification(args):
    # imports are inside so `--help` stays fast
    from src.classification.common import DEPLOYED_MODEL
    model = args.model or DEPLOYED_MODEL

    if args.command == "tune":
        from src.classification.tune import tune
        tune(model, n_iter=args.n_iter)
    elif args.command == "train":
        from src.classification.train import train
        train(model)
    elif args.command == "evaluate":
        from src.classification.evaluate import evaluate
        evaluate(args.model or "all")
    elif args.command == "predict":
        from src.inference.classification_predict import run_predict
        run_predict(model, args.json)
    elif args.command == "results":
        from src.classification.evaluate import show_results
        show_results()


# other tasks plug in here (teammates add their own function)
HANDLERS = {"classification": run_classification}


def main():
    parser = argparse.ArgumentParser(description="City Hotel ML model: tune/train/evaluate/predict")
    parser.add_argument("command", choices=["tune", "train", "evaluate", "predict", "results"])
    parser.add_argument("--task", default="classification",
                        choices=["classification", "regression", "clustering"])
    parser.add_argument("--model", choices=MODEL_CHOICES + ["all"],
                        help="model name (evaluate also accepts 'all')")
    parser.add_argument("--n-iter", type=int, help="tune only: number of random settings to try")
    parser.add_argument("--json", help="predict only: path of a JSON file with the booking details")
    args = parser.parse_args()

    if args.model == "all" and args.command != "evaluate":
        parser.error("--model all only works with evaluate")
    handler = HANDLERS.get(args.task)
    if handler is None:
        parser.error(f"task '{args.task}' is not connected to main.py yet")
    handler(args)


if __name__ == "__main__":      # needed on Windows because n_jobs=-1 starts new processes
    main()

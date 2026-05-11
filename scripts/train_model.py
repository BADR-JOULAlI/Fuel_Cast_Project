from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fuelcast import FuelCastPreprocessor


TARGET = "Consumer_Total_MomentaryFuel"


def build_model(name: str):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        )

    raise ValueError(f"Unsupported model: {name}")


def evaluate(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train FuelCast model for Lightning AI.")
    parser.add_argument("--model", choices=["random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--dataset-name", default="krohnedigital/FuelCast")
    parser.add_argument("--dataset-config", default="cps_poseidon")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", default=None, help="Example: username/fuelcast-model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from Hugging Face...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    df = dataset["train"].to_pandas()

    if args.sample_size:
        sample_size = min(args.sample_size, len(df))
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Using sample_size={sample_size}")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        shuffle=True,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", FuelCastPreprocessor()),
            ("model", build_model(args.model)),
        ]
    )

    print(f"Training {args.model}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    metrics.update(
        {
            "model": args.model,
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "target": TARGET,
        }
    )

    model_path = output_dir / "fuelcast_model.joblib"
    joblib.dump(pipeline, model_path)

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "input_schema.json", {"columns": X.columns.tolist(), "target": TARGET})
    save_json(output_dir / "sample_input.json", X_test.iloc[0].to_dict())

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importances.to_csv(output_dir / "feature_importances.csv", index=False)

    print("Training complete.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model to: {model_path}")

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--hf-repo-id is required when --push-to-hub is used")

        print(f"Uploading artifacts to Hugging Face Hub: {args.hf_repo_id}")
        api = HfApi()
        api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.hf_repo_id,
            repo_type="model",
            folder_path=str(output_dir),
            path_in_repo=".",
        )
        print("Upload complete.")


if __name__ == "__main__":
    main()

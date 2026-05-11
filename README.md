# FuelCast Training Project

Training pipeline for predicting ship fuel consumption using the
[`krohnedigital/FuelCast`](https://huggingface.co/datasets/krohnedigital/FuelCast)
dataset.

This repository is dedicated to model training, evaluation, artifact generation,
and publication to Hugging Face. The fullstack application that consumes the
published model is maintained separately.

## Links

- GitHub repository: [BADR-JOULAlI/Fuel_Cast_Project](https://github.com/BADR-JOULAlI/Fuel_Cast_Project)
- Published model: [sailu4/fuelcast-model](https://huggingface.co/sailu4/fuelcast-model)
- Dataset: [krohnedigital/FuelCast](https://huggingface.co/datasets/krohnedigital/FuelCast)

## Project Scope

This project provides:

- a cleaned notebook for exploratory analysis and model comparison;
- a reusable preprocessing pipeline that avoids target leakage;
- a training script for local or Lightning AI execution;
- model export as a `joblib` pipeline;
- optional upload of model artifacts to Hugging Face Hub.

The target variable is:

```text
Consumer_Total_MomentaryFuel
```

To keep the evaluation realistic, fuel-related leakage columns such as
`MomentaryFuel` features and `Total_Engine_Fuel` are excluded from the model
inputs.

## Repository Structure

```text
Fuel_Cast_Project.ipynb        Exploratory analysis and model comparison
src/fuelcast/                  Reusable preprocessing code
scripts/train_model.py         Training, evaluation, export, and HF upload
lightning_ai/                  Lightning AI execution notes
requirements.txt               Python dependencies
```

Generated artifacts are saved under `artifacts/` and are intentionally ignored
by Git.

## Latest Local Training Result

Sample run with `RandomForestRegressor` on `30,000` rows:

| Metric | Value |
| --- | ---: |
| R2 | 0.9957 |
| RMSE | 0.0295 |
| MAE | 0.0194 |

Rows used:

```text
train_rows = 24,000
test_rows  = 6,000
```

## Installation

```bash
pip install -r requirements.txt
```

## Train Locally

Recommended first run with a smaller sample:

```bash
python scripts/train_model.py \
  --model random_forest \
  --sample-size 30000 \
  --output-dir artifacts
```

Full training run:

```bash
python scripts/train_model.py \
  --model random_forest \
  --output-dir artifacts
```

Available model choices:

```text
random_forest
gradient_boosting
```

## Generated Artifacts

After training, the script writes:

```text
artifacts/fuelcast_model.joblib
artifacts/metrics.json
artifacts/input_schema.json
artifacts/sample_input.json
artifacts/feature_importances.csv
artifacts/README.md
```

The `fuelcast_model.joblib` file contains the complete scikit-learn pipeline:
preprocessing plus trained estimator.

## Publish to Hugging Face

Log in with a Hugging Face token that has **Write** permission:

```bash
hf auth login
```

Then train and upload:

```bash
python scripts/train_model.py \
  --model random_forest \
  --sample-size 30000 \
  --output-dir artifacts \
  --push-to-hub \
  --hf-repo-id sailu4/fuelcast-model
```

Published model:

```text
https://huggingface.co/sailu4/fuelcast-model
```

## Lightning AI Workflow

In Lightning AI Studio:

```bash
git clone https://github.com/BADR-JOULAlI/Fuel_Cast_Project.git
cd Fuel_Cast_Project
pip install -r requirements.txt
python scripts/train_model.py --model random_forest --sample-size 30000 --output-dir artifacts
```

To publish from Lightning AI, configure `HF_TOKEN` or run `hf auth login`, then
add:

```bash
--push-to-hub --hf-repo-id sailu4/fuelcast-model
```

## Fullstack App Integration

The fullstack app should load the model from Hugging Face using:

```bash
HF_MODEL_REPO=sailu4/fuelcast-model
```

Local fullstack project path:

```text
C:\Users\anony\Downloads\Fuel_Cast_Fullstack
```

## Notes

- The notebook is for analysis and reporting.
- The training script is the production path for creating the model artifact.
- Hugging Face upload requires a valid token with write access.
- The model repo may show both "Published a model" and "Updated a model" in
  recent activity; that is normal when the repo is created and then artifacts
  are uploaded.

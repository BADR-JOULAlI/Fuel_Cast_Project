from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: dict[str, Any]


def resolve_model_path() -> Path:
    local_path = os.getenv("FUELCAST_MODEL_PATH", "artifacts/fuelcast_model.joblib")
    if Path(local_path).exists():
        return Path(local_path)

    repo_id = os.getenv("HF_MODEL_REPO")
    if repo_id:
        return Path(hf_hub_download(repo_id=repo_id, filename="fuelcast_model.joblib"))

    raise FileNotFoundError(
        "Model not found. Set FUELCAST_MODEL_PATH or HF_MODEL_REPO, "
        "or run scripts/train_model.py first."
    )


app = FastAPI(title="FuelCast Prediction API", version="1.0.0")
pipeline = None

frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.on_event("startup")
def load_model():
    global pipeline
    model_path = resolve_model_path()
    pipeline = joblib.load(model_path)


@app.get("/")
def index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "FuelCast API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(payload: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    frame = pd.DataFrame([payload.features])
    prediction = float(pipeline.predict(frame)[0])
    return {"predicted_fuel_consumption": prediction}

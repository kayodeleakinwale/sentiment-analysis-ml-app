from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODEL_PATH = REPO_ROOT / "models" / "sentiment_pipeline.joblib"

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")

class PredictResponse(BaseModel):
    label: str
    score: float | None = None  # may be None depending on model
    raw: Dict[str, Any] | None = None

pipeline = None

@app.on_event("startup")
def load_model() -> None:
    global pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run training first: python src/train.py --data data/sample.csv"
        )
    pipeline = joblib.load(MODEL_PATH)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Predict class (0/1)
    pred = int(pipeline.predict([req.text])[0])
    label = "positive" if pred == 1 else "negative"

    score = None
    raw = None

    # If model supports probabilities, return probability of positive class
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([req.text])[0]
        # proba index 1 is class "1" probability in typical sklearn setups
        score = float(proba[1])
        raw = {"proba_negative": float(proba[0]), "proba_positive": float(proba[1])}

    return PredictResponse(label=label, score=score, raw=raw)


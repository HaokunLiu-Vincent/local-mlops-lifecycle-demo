import numpy as np
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "release" / "model"

app = FastAPI(title="DNN Model Serving Demo")

# Load model once during startup, not per request
model = mlflow.pyfunc.load_model(str(MODEL_PATH))

FEATURES = [f"f{i}" for i in range(8)]


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        min_length=8,
        max_length=8,
    )


class PredictionResponse(BaseModel):
    probability: float
    predicted_class: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    df = pd.DataFrame(
        [request.features],
        columns=FEATURES,
        dtype=np.float32,
    )

    probability = float(model.predict(df)[0])

    return PredictionResponse(
        probability=probability,
        predicted_class=int(probability >= 0.5),
    )

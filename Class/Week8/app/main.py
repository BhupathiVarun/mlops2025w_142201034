# app/main.py
import os
import joblib
import wandb
from fastapi import FastAPI, Request, HTTPException
import numpy as np
from typing import List, Any

app = FastAPI()

# default artifact reference (can override with env var)
MODEL_ARTIFACT = os.environ.get(
    "WANDB_MODEL_ARTIFACT",
    "142201034-indian-institute-of-technology/classroom-deploy2/iris-rf:latest",
)

def load_model_from_wandb(artifact_ref: str):
    """Download model artifact from wandb and load with joblib."""
    try:
        wandb.login()
    except Exception:
        # ignore login errors (e.g., running locally without interactive login)
        pass

    print(f"Attempting to load model from: {artifact_ref}")
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    path = artifact.download()
    model_file = os.path.join(path, "model.pkl")
    print("Model downloaded and loaded successfully.")
    return joblib.load(model_file)

@app.on_event("startup")
def startup():
    global model
    model = load_model_from_wandb(MODEL_ARTIFACT)

@app.get("/")
def root():
    return {"status": "ok", "model_artifact": MODEL_ARTIFACT}

@app.post("/predict")
async def predict(request: Request):
    """
    Accepts either:
    - a JSON list: [5, 3.5, 1.4, 0.2]
    - or a JSON object: {"features": [5, 3.5, 1.4, 0.2]}
    Returns {"prediction": int}
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Log raw body for debugging (will appear in uvicorn console)
    print("raw request body:", body)

    # normalize possible shapes
    if isinstance(body, list):
        features = body
    elif isinstance(body, dict) and "features" in body:
        features = body["features"]
    else:
        # Provide informative error detail for client
        raise HTTPException(
            status_code=422,
            detail="Request body must be either a JSON list (e.g. [5,3.5,...]) "
                   "or an object with a 'features' key (e.g. {\"features\": [...]})"
        )

    # Validate features is a list of numbers
    if not isinstance(features, list) or len(features) == 0:
        raise HTTPException(status_code=422, detail="`features` must be a non-empty list.")

    try:
        arr = np.asarray(features, dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not convert features to numeric array: {e}")

    try:
        pred = model.predict(arr)
        return {"prediction": int(pred[0])}
    except Exception as e:
        # If prediction fails, return a 500-level error with message
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

import os
import re
import shutil
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import predict
from utils import get_feature_names

# Base setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Response Models
class PredictionResponse(BaseModel):
    label: str
    probability: float
    per_epoch: List[float]
    n_epochs: int

class RetrainResponse(BaseModel):
    status: str
    label: str

class FeatureScore(BaseModel):
    name: str
    importance: float

class FeatureImportanceResponse(BaseModel):
    features: List[FeatureScore]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# App Setup
app = FastAPI(
    title="EEG Stress Classifier",
    description="API for EEG stress detection, retraining and feature importance",
    version="1.0.0",
    openapi_tags=[
        {"name": "Health", "description": "API status"},
        {"name": "Prediction", "description": "Stress prediction from EEG"},
        {"name": "Retraining", "description": "Model retraining with new data"},
        {"name": "Feature Importance", "description": "Model feature importance scores"}
    ]
)

# CORS - Note: Change ["*"] to specific domain origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Temp Upload Folder
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Static files setup with path check
WEB_DIR = os.path.join(BASE_DIR, "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# Health
@app.get("/health", tags=["Health"], response_model=HealthResponse)
def health():
    return {
        "status": "running",
        "model_loaded": predict.model is not None
    }


# Predict
@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".edf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only .edf files are accepted."
        )

    # Isolated temp file creation to avoid collisions under concurrent load
    safe_filename = re.sub(r'[^\w\-.]', '_', os.path.basename(file.filename))
    
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=f"_{safe_filename}") as temp_file:
        temp_path = temp_file.name
        # Stream file to disk directly to prevent high memory usage
        shutil.copyfileobj(file.file, temp_file)

    try:
        # Check file size after writing stream
        if os.path.getsize(temp_path) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        result = predict.predict_stress(temp_path)
        return result

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during prediction: {e}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Retrain
@app.post("/retrain", tags=["Retraining"], response_model=RetrainResponse)
async def retrain_endpoint(
    file: UploadFile = File(...),
    label: int = Form(...)
):
    if not file.filename or not file.filename.lower().endswith(".edf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only .edf files are accepted."
        )

    if label not in [0, 1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid label. Must be 0 (Calm) or 1 (Stress)."
        )

    safe_filename = re.sub(r'[^\w\-.]', '_', os.path.basename(file.filename))
    
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=f"_{safe_filename}") as temp_file:
        temp_path = temp_file.name
        shutil.copyfileobj(file.file, temp_file)

    try:
        if os.path.getsize(temp_path) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        predict.retrain(temp_path, true_label=label)
        return {
            "status": "retrained",
            "label": "Stress" if label == 1 else "Calm"
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during retraining: {e}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Feature Importance
@app.get("/feature-importance", tags=["Feature Importance"], response_model=FeatureImportanceResponse)
def feature_importance():
    try:
        if predict.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded."
            )

        first_layer_weights = predict.model.layers[0].get_weights()
        if not first_layer_weights:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model has no weights. It may not have been trained yet."
            )

        importance = np.abs(first_layer_weights[0]).mean(axis=1)
        importance = importance / importance.sum()

        feature_names = get_feature_names()

        if len(feature_names) != len(importance):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Mismatch between feature names and model weights."
            )

        ranked = sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "features": [
                {"name": name, "importance": round(score, 4)}
                for name, score in ranked
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error fetching feature importance: {e}"
        )


# Serve Web UI
@app.get("/")
def serve_ui():
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Web UI not found."
        )
    return FileResponse(index_path)


# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
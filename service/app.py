import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn
import pandas as pd

# Model URI pointing to the champion alias registered in MLflow
MODEL_URI = os.getenv("MODEL_URI", "models:/BikeDemandChampion@champion")
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        # Load the registered champion model directly from MLflow registry/artifacts
        model = mlflow.sklearn.load_model(MODEL_URI)
        print(f"[INFO] Successfully loaded model from '{MODEL_URI}'")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        model = None
    yield

app = FastAPI(
    title="Bike Demand Prediction Service",
    description="FastAPI service hosting the MLflow @champion model for real-time inference.",
    version="1.0.0",
    lifespan=lifespan
)

class BikeFeatures(BaseModel):
    season: int = Field(..., ge=1, le=4, description="1: Spring, 2: Summer, 3: Fall, 4: Winter")
    yr: int = Field(..., ge=0, le=1, description="0: 2011, 1: 2012")
    mnth: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    hr: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    holiday: int = Field(..., ge=0, le=1, description="1: Holiday, 0: Not holiday")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0: Sun to 6: Sat)")
    workingday: int = Field(..., ge=0, le=1, description="1: Working day, 0: Weekend/Holiday")
    weathersit: int = Field(..., ge=1, le=4, description="1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain/Snow")
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature")
    atemp: float = Field(..., ge=0.0, le=1.0, description="Normalized feels-like temperature")
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized humidity")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized windspeed")

@app.get("/")
def root():
    return {
        "message": "Bike Demand API is running.",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI
    }

@app.post("/predict")
def predict_demand(features: BikeFeatures):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Prediction model is not initialized. Please verify MLflow registry."
        )
    
    # Structure input data to match exact feature names and float types
    input_data = pd.DataFrame([{
        "season": float(features.season),
        "yr": float(features.yr),
        "mnth": float(features.mnth),
        "hr": float(features.hr),
        "holiday": float(features.holiday),
        "weekday": float(features.weekday),
        "workingday": float(features.workingday),
        "weathersit": float(features.weathersit),
        "temp": float(features.temp),
        "atemp": float(features.atemp),
        "hum": float(features.hum),
        "windspeed": float(features.windspeed)
    }])
    
    try:
        prediction = model.predict(input_data)[0]
        # Demand cannot be negative, rounded to nearest whole bike count
        predicted_count = int(max(0, round(prediction)))
        return {
            "predicted_demand": predicted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
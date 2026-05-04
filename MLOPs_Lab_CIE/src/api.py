from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
from datetime import datetime
import os

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")

# Input validation
class InputData(BaseModel):
    pipe_age_years: int = Field(..., ge=1, le=100)
    flow_rate_lph: float = Field(..., ge=100, le=10000)
    moisture_pct: float = Field(..., ge=0, le=200)
    wall_thickness_mm: float = Field(..., ge=1, le=20)

# Health endpoint
@app.get("/heartbeat")
def heartbeat():
    return {
        "alive": True,
        "service": "PipeWatch corrosion_risk_score API"
    }

# Prediction endpoint
@app.post("/infer")
def infer(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]

    # 🔥 Logging (needed for Task 4)
    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": data.dict(),
        "prediction": float(pred)
    }

    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": float(pred)}
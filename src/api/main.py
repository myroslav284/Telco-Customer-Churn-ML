import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import gradio as gr
import joblib
import pandas as pd

from src.api.schemas import CustomerData, PredictionResponse

THRESHOLD = 0.3
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = project_root / "src" / "models" / "xgb_pipeline.pkl"
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}")
    pipeline = joblib.load(model_path)
    yield
    pipeline = None

# --- FASTAPI APP ---
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="High-performance ML serving with XGBoost Pipeline and Gradio UI",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerData):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    input_df = pd.DataFrame([customer.model_dump()])
    proba = float(pipeline.predict_proba(input_df)[:, 1][0])
    prediction = int(proba >= THRESHOLD)
    status_label = "High Risk (Likely to churn)" if prediction == 1 else "Low Risk (Not likely to churn)"
    
    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(proba, 4),
        status=status_label
    )

# --- GRADIO WEB INTERFACE ---
def run_ui_prediction(
    gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    input_df = pd.DataFrame([payload])
    proba = float(pipeline.predict_proba(input_df)[:, 1][0])
    label = "🚨 LIKELY TO CHURN" if proba >= THRESHOLD else "✅ LOYAL CUSTOMER"
    return f"{label}\nChurn Probability: {proba:.2%}"

demo = gr.Interface(
    fn=run_ui_prediction,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Female"),
        gr.Dropdown(["No", "Yes"], label="Senior Citizen", value="No"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method", value="Electronic check"),
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=10000),
    ],
    outputs=gr.Textbox(label="Result", lines=2),
    title="🔮 Telco Customer Churn Predictor",
    description="Interactive ML application powered by FastAPI, XGBoost, and Gradio.",
    examples=[
        ["Female", "No", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 1, 85.0, 85.0],
        ["Male", "No", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No", "Two year", "No", "Credit card (automatic)", 60, 45.0, 2700.0]
    ],
    theme=gr.themes.Soft()
)

app = gr.mount_gradio_app(app, demo, path="/ui")
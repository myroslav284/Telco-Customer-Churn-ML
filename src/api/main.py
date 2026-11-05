from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import shap
from src.api.agent import generate_retention_plan
from src.api.schemas import CustomerData, FactorImpact, PredictionResponse

load_dotenv()

THRESHOLD = 0.3
pipeline = None
explainer = None
feature_names = None


def extract_shap_factors(
    input_df: pd.DataFrame, top_k: int = 3
) -> Tuple[List[Dict], List[Dict]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_data = preprocessor.transform(input_df)

    raw_shap = explainer.shap_values(transformed_data)
    sample_shap = raw_shap[0] if len(raw_shap.shape) == 2 else raw_shap

    dense_vals = (
        transformed_data.toarray()[0]
        if hasattr(transformed_data, "toarray")
        else transformed_data[0]
    )

    impacts = []
    for name, val, actual_val in zip(feature_names, sample_shap, dense_vals):
        if actual_val == 0:
            continue

        clean_name = name.split("__")[-1]
        impacts.append({"feature": clean_name, "impact": round(float(val), 4)})

    risk_drivers = sorted(
        [i for i in impacts if i["impact"] > 0],
        key=lambda x: x["impact"],
        reverse=True,
    )[:top_k]

    retention_drivers = sorted(
        [i for i in impacts if i["impact"] < 0], key=lambda x: x["impact"]
    )[:top_k]

    return risk_drivers, retention_drivers


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, explainer, feature_names
    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = project_root / "src" / "models" / "xgb_pipeline.pkl"
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}")

    pipeline = joblib.load(model_path)

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    explainer = shap.TreeExplainer(model)
    feature_names = preprocessor.get_feature_names_out()

    yield

    pipeline = None
    explainer = None
    feature_names = None


# --- FASTAPI APP ---
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "High-performance ML serving with XGBoost, SHAP Explainability and Gradio UI"
    ),
    version="1.2.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerData):
    if pipeline is None or explainer is None:
        raise HTTPException(status_code=503, detail="Pipeline or Explainer not ready")

    input_df = pd.DataFrame([customer.model_dump()])
    proba = float(pipeline.predict_proba(input_df)[:, 1][0])
    prediction = int(proba >= THRESHOLD)
    status_label = (
        "High Risk (Likely to churn)"
        if prediction == 1
        else "Low Risk (Not likely to churn)"
    )

    risk_drivers, retention_drivers = extract_shap_factors(input_df)

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(proba, 4),
        status=status_label,
        top_risk_drivers=[FactorImpact(**item) for item in risk_drivers],
        top_retention_drivers=[FactorImpact(**item) for item in retention_drivers],
    )


# --- GRADIO WEB INTERFACE (English, Minimalist) ---
def run_ui_prediction(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges,
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
    label = "🚨 HIGH RISK OF CHURN" if proba >= THRESHOLD else "✅ LOYAL CUSTOMER"

    risk_drivers, retention_drivers = extract_shap_factors(input_df)

    result_text = f"## {label}\n"
    result_text += f"**Churn Probability:** `{proba:.2%}`\n\n---\n"

    result_text += "### 🔍 Top Risk Factors (Increasing Churn):\n"
    if risk_drivers:
        for item in risk_drivers:
            result_text += f"- **{item['feature']}**: `+{item['impact']:.3f}`\n"
    else:
        result_text += "_None identified_\n"

    result_text += "\n### 🛡️ Top Retention Factors (Protecting from Churn):\n"
    if retention_drivers:
        for item in retention_drivers:
            result_text += f"- **{item['feature']}**: `{item['impact']:.3f}`\n"
    else:
        result_text += "_None identified_\n"

    if proba >= THRESHOLD:
        agent_plan = generate_retention_plan(
            churn_prob=proba,
            customer_profile=payload,
            top_risks=risk_drivers,
            top_protections=retention_drivers,
        )
    else:
        agent_plan = "✅ **Account in Good Standing**: Retention offer generation is not triggered for low-risk customers."

    return result_text, agent_plan


def run_shap_counterfactual_simulation(
    gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    tenure, MonthlyCharges, TotalCharges,
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
    orig_proba = float(pipeline.predict_proba(input_df)[:, 1][0])

    if orig_proba < THRESHOLD:
        return "✅ **Account in Good Standing**: Customer is already at low churn risk. No countermeasures required."

    risk_drivers, _ = extract_shap_factors(input_df)

    sim_payload = payload.copy()
    applied_actions = []

    for item in risk_drivers:
        feat = item["feature"]
        if "Contract" in feat and sim_payload["Contract"] == "Month-to-month":
            sim_payload["Contract"] = "One year"
            applied_actions.append(f"**{feat}** ➔ Switched from Month-to-month to **One year contract**")
        elif "OnlineSecurity" in feat and sim_payload["OnlineSecurity"] != "Yes":
            sim_payload["OnlineSecurity"] = "Yes"
            applied_actions.append(f"**{feat}** ➔ Activated complimentary **Online Security**")
        elif "TechSupport" in feat and sim_payload["TechSupport"] != "Yes":
            sim_payload["TechSupport"] = "Yes"
            applied_actions.append(f"**{feat}** ➔ Activated complimentary **Tech Support**")
        elif "Electronic check" in feat:
            sim_payload["PaymentMethod"] = "Bank transfer (automatic)"
            applied_actions.append(f"**{feat}** ➔ Migrated to **Bank transfer (automatic)**")

    has_price_or_tenure = any("tenure" in d["feature"] or "Charges" in d["feature"] for d in risk_drivers)
    if has_price_or_tenure and sim_payload["MonthlyCharges"] > 30:
        old_charge = sim_payload["MonthlyCharges"]
        new_charge = round(old_charge * 0.85, 2)
        sim_payload["MonthlyCharges"] = new_charge
        applied_actions.append(f"**Tenure / Spend sensitivity** ➔ Applied **15% discount** (${old_charge} ➔ ${new_charge}/mo)")

    new_proba = float(pipeline.predict_proba(pd.DataFrame([sim_payload]))[:, 1][0])
    delta = new_proba - orig_proba
    status = "✅ **Successfully shifted to Safe / Loyal Zone!**" if new_proba < THRESHOLD else "⚠️ **Risk reduced, but requires further action.**"

    actions_md = "\n".join([f"- {a}" for a in applied_actions]) if applied_actions else "- No actionable SHAP drivers found to alter."

    return f"""### 🧪 Counterfactual Simulation (SHAP Auto-Fix)
- **Original Churn Risk:** `{orig_proba:.2%}` 🚨
- **Simulated Churn Risk:** `{new_proba:.2%}` ✅
- **Risk Delta:** `{delta:+.2%}` 📉

**Automated Remediation Actions:**
{actions_md}

**Impact Status:** {status}
"""

with gr.Blocks(title="Customer Churn AI") as demo:
    gr.Markdown(
        """
        # 🔮 Telco Customer Churn & Explainability
        Predict churn probability, inspect local SHAP factor contributions, and generate AI retention strategies.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("📄 Contract & Billing"):
                    with gr.Row():
                        contract = gr.Dropdown(
                            ["Month-to-month", "One year", "Two year"],
                            label="Contract",
                            value="Month-to-month",
                        )
                        payment_method = gr.Dropdown(
                            [
                                "Electronic check",
                                "Mailed check",
                                "Bank transfer (automatic)",
                                "Credit card (automatic)",
                            ],
                            label="Payment Method",
                            value="Electronic check",
                        )
                    with gr.Row():
                        tenure = gr.Slider(
                            0, 72, value=1, step=1, label="Tenure (Months)"
                        )
                        monthly_charges = gr.Number(
                            value=85.0, label="Monthly Charges ($)"
                        )
                        total_charges = gr.Number(value=85.0, label="Total Charges ($)")
                    with gr.Row():
                        paperless = gr.Radio(
                            ["Yes", "No"], label="Paperless Billing", value="Yes"
                        )

                with gr.Tab("🌐 Internet & Services"):
                    with gr.Row():
                        internet = gr.Dropdown(
                            ["Fiber optic", "DSL", "No"],
                            label="Internet Service",
                            value="Fiber optic",
                        )
                        online_sec = gr.Dropdown(
                            ["No", "Yes", "No internet service"],
                            label="Online Security",
                            value="No",
                        )
                        tech_supp = gr.Dropdown(
                            ["No", "Yes", "No internet service"],
                            label="Tech Support",
                            value="No",
                        )
                    with gr.Row():
                        backup = gr.Dropdown(
                            ["No", "Yes", "No internet service"],
                            label="Online Backup",
                            value="No",
                        )
                        device_prot = gr.Dropdown(
                            ["No", "Yes", "No internet service"],
                            label="Device Protection",
                            value="No",
                        )
                    with gr.Row():
                        stream_tv = gr.Dropdown(
                            ["No", "Yes", "No internet service"],
                            label="Streaming TV",
                            value="Yes",
                        )
                        stream_mov = gr.Dropdown(
                            ["No", "Yes", "No internet service"],
                            label="Streaming Movies",
                            value="Yes",
                        )

                with gr.Tab("👤 Demographics"):
                    with gr.Row():
                        gender = gr.Radio(
                            ["Female", "Male"], label="Gender", value="Female"
                        )
                        senior = gr.Radio(
                            ["No", "Yes"], label="Senior Citizen", value="No"
                        )
                    with gr.Row():
                        partner = gr.Radio(["No", "Yes"], label="Partner", value="No")
                        dependents = gr.Radio(
                            ["No", "Yes"], label="Dependents", value="No"
                        )
                    with gr.Row():
                        phone = gr.Radio(
                            ["Yes", "No"], label="Phone Service", value="Yes"
                        )
                        multiple_lines = gr.Dropdown(
                            ["No", "Yes", "No phone service"],
                            label="Multiple Lines",
                            value="No",
                        )

            submit_btn = gr.Button(
                "Analyze Churn Risk & Generate Plan",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=2):
            output_markdown = gr.Markdown(
                "*Click **Analyze Churn Risk & Generate Plan** to inspect factor impacts.*"
            )
            gr.Markdown("---")
            agent_markdown = gr.Markdown(
                "*Retention recommendations will appear here automatically if the risk threshold is reached.*"
            )

            gr.Markdown("---")
            sim_btn = gr.Button("  Simulate SHAP Countermeasures", variant="secondary")
            sim_output = gr.Markdown("*Click above to mathematically simulate resolving the top SHAP drivers.*")

    submit_btn.click(
        fn=run_ui_prediction,
        inputs=[
            gender,
            senior,
            partner,
            dependents,
            phone,
            multiple_lines,
            internet,
            online_sec,
            backup,
            device_prot,
            tech_supp,
            stream_tv,
            stream_mov,
            contract,
            paperless,
            payment_method,
            tenure,
            monthly_charges,
            total_charges,
        ],
        outputs=[output_markdown, agent_markdown],
    )

    sim_btn.click(
        fn=run_shap_counterfactual_simulation,
        inputs=[
            gender, senior, partner, dependents, phone, multiple_lines,
            internet, online_sec, backup, device_prot, tech_supp,
            stream_tv, stream_mov, contract, paperless, payment_method,
            tenure, monthly_charges, total_charges,
        ],
        outputs=sim_output,
    )

app = gr.mount_gradio_app(app, demo, path="/ui")

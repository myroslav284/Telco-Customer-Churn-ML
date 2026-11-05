# Telco Customer Churn Prediction: MLOps, Explainability & AI Retention

[![CI/CD Pipeline](https://github.com/myroslav284/Telco-Customer-Churn-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/myroslav284/Telco-Customer-Churn-ML/actions/workflows/ci.yml)

## 📌 Project Overview
This project is an end-to-end production-grade Machine Learning and Explainable AI (XAI) system for predicting and preventing customer churn in telecommunications. It integrates an optimized **XGBoost** classifier, local factor explainability via **SHAP**, a **Groq LLM-powered Retention Agent**, and an interactive **Counterfactual "What-If" Simulator**.

The system is deployed via **FastAPI** and **Gradio**, fully containerized with **Docker**, and continuously deployed to **Azure Container Apps (ACA)** using GitHub Actions.

---

## 🏗️ Project Architecture & Components

```text
Telco-Customer-Churn-ML/
├── .github/workflows/ci.yml      # CI/CD: Automated Docker build & Azure Container Apps deployment
├── data/raw/                     # Raw Kaggle Telco dataset (gitignored)
├── notebooks/
│   └── 01_model_training.ipynb   # EDA, baseline comparison, Optuna tuning, MLflow tracking
├── src/
│   ├── api/
│   │   ├── agent.py              # Groq LLM Retention Specialist agent
│   │   ├── main.py               # FastAPI app, SHAP calculation & Gradio UI
│   │   └── schemas.py            # Pydantic validation schemas
│   └── models/
│       └── xgb_pipeline.pkl      # Production Scikit-Learn + XGBoost pipeline
├── Dockerfile                    # Containerization for production deployment
├── requirements.txt              # Pinned production dependencies
└── README.md
```

---

## 🚀 Key Features

* **Predictive ML (XGBoost):** Optimized classification pipeline maximizing Recall on churning customers to catch at-risk accounts early.
* **Explainable AI (SHAP):** Computes exact positive impacts (increasing churn) vs negative impacts (protecting retention) using `TreeExplainer` for every single customer evaluation.
* **AI Retention Specialist (Groq LLM):** In the Gradio UI, an LLM agent combines customer tenure, billing metadata, and SHAP contributors to generate a risk diagnosis and 3 prioritized retention actions.
* **Counterfactual What-If Simulation:** A 1-click automated simulator within the UI that mathematically applies counter-measures to the customer's top SHAP drivers (e.g. converting Month-to-month contracts, adding security) and recalculates the new churn probability instantly (~0.01s).
* **Production REST API:** High-performance, lightweight FastAPI endpoints returning pure ML predictions and SHAP factor breakdowns for CRM integrations.
* **Cloud-Native CI/CD:** GitHub Actions workflow building multi-tag container images pushed to GitHub Container Registry (GHCR) and deployed to **Azure Container Apps (ACA)**.

---

## 🛠️ Technology Stack

* **ML & Explainability:** Scikit-Learn, XGBoost, LightGBM, SHAP (`TreeExplainer`)
* **Generative AI:** Groq Cloud API (`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`)
* **API & Serving:** FastAPI, Uvicorn, Pydantic v2
* **Web Interface:** Gradio Blocks
* **Tracking & Tuning:** Optuna, MLflow
* **Cloud & DevOps:** Docker, GitHub Container Registry (GHCR), Microsoft Azure Container Apps (ACA), uv

---

## 💻 Setup & Installation (Local Development)

This project uses the modern **uv** package manager.

### 1. Clone the repository:
```bash
git clone https://github.com/myroslav284/Telco-Customer-Churn-ML.git
cd Telco-Customer-Churn-ML
```

### 2. Configure Environment Variables:
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

### 3. Create Virtual Environment & Install Dependencies:
```bash
uv venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

uv pip install -r requirements.txt
```

### 4. Run the Application:
```bash
uvicorn src.api.main:app --reload
```
Once the server is running, access:
* **Interactive Web Dashboard:** [http://localhost:8000/ui](http://localhost:8000/ui)
* **REST API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check Endpoint:** [http://localhost:8000/](http://localhost:8000/)

---

## 📡 REST API Usage

### `POST /predict`
FastAPI provides high-throughput ML inference and SHAP factor breakdowns:

**Request:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 85.0
}
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.931,
  "status": "High Risk (Likely to churn)",
  "top_risk_drivers": [
    {"feature": "tenure", "impact": 0.819},
    {"feature": "Contract_Month-to-month", "impact": 0.529},
    {"feature": "OnlineSecurity_No", "impact": 0.224}
  ],
  "top_retention_drivers": [
    {"feature": "MultipleLines_No", "impact": -0.112},
    {"feature": "SeniorCitizen_0", "impact": -0.009},
    {"feature": "gender_Female", "impact": -0.002}
  ]
}
```

---

## 🐳 Docker & Cloud Deployment

### Local Docker Build & Run:
```bash
# Build Docker image
docker build -t telco-churn-api .

# Run container with environment file
docker run -p 8000:8000 --env-file .env telco-churn-api
```

### Automated Azure Deployment:
Every push to `main` triggers `.github/workflows/ci.yml`:
1. Builds the Docker container.
2. Pushes immutable SHA-tagged images to `ghcr.io/myroslav284/telco-customer-churn-ml`.
3. Updates the **Azure Container App** (`telco-churn-api`) in the `churn-ml-rg` resource group.

---

## 📊 Model Performance Highlights

Evaluated on the Telco Customer Churn test split at threshold `0.30`:

* **Production Model:** XGBoost (Optuna Tuned Pipeline)
* **Optimization Goal:** Prioritize Recall on churning accounts for proactive retention
* **Final Recall (Threshold 0.30):** `0.925` (capturing 92.5% of all churning customers)
* **Test ROC-AUC:** `0.8475`
* **Generalization Gap (Train AUC - Test AUC):** `0.0169` (indicating strong resistance to overfitting)

> *Note on Model Training:* The production model is pre-serialized in `src/models/xgb_pipeline.pkl`. If you wish to retrain the pipeline via `notebooks/01_model_training.ipynb`, download `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/raw/` (excluded from git by design).

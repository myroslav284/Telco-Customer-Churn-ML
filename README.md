# Telco Customer Churn Prediction: MLOps

## Project Overview
This project focuses on predicting customer churn in the telecommunications industry using a high-performance **XGBoost** machine learning model. The project encompasses a full MLOps lifecycle: from exploratory data analysis (EDA) and model training to deploying a REST API using **FastAPI** and an interactive **Gradio** web interface. Key emphasis is placed on hyperparameter optimization via **Optuna** and experiment tracking/model serialization using **MLflow**.

## Key Features
*   **Data Analysis & Preprocessing:** Handling class imbalance, missing values, and multicollinearity utilizing `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`.
*   **Model Selection & Tuning:** Comparative analysis of baseline models (Logistic Regression, Random Forest, LightGBM, XGBoost) and advanced hyperparameter optimization for XGBoost using Optuna. The objective is to maximize Recall while minimizing overfitting (ROC-AUC analysis).
*   **MLOps Tracking:** Integration of MLflow for logging parameters, metrics (Recall, Precision, F1, ROC-AUC), and exporting the serialized `xgb_pipeline.pkl` artifact.
*   **Production API:** Exposing the trained model via a FastAPI REST endpoint with strict data validation utilizing Pydantic schemas.
*   **Interactive UI:** Gradio web application for seamless interaction and prediction visualization.
*   **Containerization:** Fully Dockerized application, ready for deployment on cloud platforms such as Azure Container Instances (ACI).

## Technology Stack
*   **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
*   **Optimization & Tracking:** Optuna, MLflow
*   **Backend & API:** FastAPI, Uvicorn, Pydantic
*   **Web Interface:** Gradio
*   **Data Manipulation:** Pandas, NumPy
*   **Deployment:** Docker, Microsoft Azure
*   **Package Management:** uv

## Setup & Installation (Local Development)

This project utilizes the **uv** package manager.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/myroslav284/Telco-Customer-Churn-ML.git
    cd Telco-Customer-Churn-ML
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate  # For Linux/macOS
    # .venv\Scripts ctivate   # For Windows
    uv pip install -r requirements.txt
    ```
3.  **Model Training (Optional):**
    Execute the Jupyter Notebook `notebooks/01_model_training.ipynb` to explore the data, train the model, and generate the `xgb_pipeline.pkl` artifact (saved in the `src/models/` directory).
4.  **Start API & UI:**
    ```bash
    uvicorn src.api.main:app --reload
    ```
    *   API Documentation (Swagger UI): `http://localhost:8000/docs`
    *   Gradio Web Interface: `http://localhost:8000/ui`

## Docker Deployment

1.  **Build the Docker image:**
    ```bash
    docker build -t telco-churn-api .
    ```
2.  **Run the container:**
    ```bash
    docker run -p 8000:8000 telco-churn-api
    ```
The service will be accessible on port 8000.

## Model Performance Highlights
*   **Model Selected:** XGBoost (Optuna Tuned)
*   **Objective:** Maximize Recall for churning customers (for proactive retention) while maintaining high overall model discriminative power.
*   **Final Recall (Threshold 0.3):** 0.925
*   **Test ROC-AUC:** 0.8475
*   **Overfitting Gap (Train - Test):** 0.0169 (Demonstrating excellent generalization capability due to robust regularization).
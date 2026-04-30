# Technical Documentation: Telco Churn – End-to-End ML Project

## 1. Project Overview

### Purpose
The purpose of this project is to build and ship a full machine-learning solution for predicting customer churn in a telecom setting. This covers everything from data preparation and modeling to exposing an API and a web UI deployed using Docker.

### Problem Solved & Benefits
- **Faster decisions:** Predicts which customers are likely to churn so teams can act before they leave.
- **Operationalized ML:** The model is accessible via a REST API and a user-friendly UI; anyone can test it without needing Jupyter notebooks.
- **Repeatable delivery:** Containerization using Docker enables every change to be built, tested, and run in a consistent way.
- **Traceable experiments:** MLflow tracks runs, metrics, parameters, and artifacts for reproducibility and auditing.

---

## 2. System Architecture

The architecture consists of several key components that work seamlessly together from training to deployment.

### Components
1. **Data & Modeling Pipeline**: Built using Python and pandas, it performs data validation (using Great Expectations), preprocessing, and deterministic feature engineering. The core model is an XGBoost classifier.
2. **Model Tracking (MLflow)**: During training, experiments (hyperparameters, precision/recall metrics) and the serialized model artifacts (along with feature mappings) are logged.
3. **Inference Service (FastAPI)**: A production-ready REST API service that loads the model via `mlflow.pyfunc` and exposes endpoints (`/` for health check, `/predict` for inferences).
4. **Web UI (Gradio)**: A user-friendly Gradio interface mounted directly onto the FastAPI application at `/ui` to allow manual testing and demonstrations.
5. **Containerization**: Everything is packaged into a Docker image, utilizing a lightweight Python 3.11-slim base, and served via `uvicorn`.

---

## 3. Technology Stack

- **Language:** Python 3.11
- **Machine Learning:** XGBoost, Scikit-learn, Optuna (for hyperparameter tuning)
- **Model Registry & Tracking:** MLflow
- **API Framework:** FastAPI, Pydantic (for request schema validation)
- **Web UI:** Gradio
- **Data Manipulation:** pandas, numpy
- **Containerization:** Docker
- **Server/ASGI:** Uvicorn

---

## 4. Project Structure

```
├── .github/          # CI/CD Workflows
├── data/             # Raw and processed datasets
├── dockerfile        # Docker build instructions
├── notebooks/        # Jupyter notebooks (e.g., EDA.ipynb)
├── requirements.txt  # Python dependencies
├── scripts/          # Execution scripts
│   ├── run_pipeline.py            # Main training pipeline execution
│   └── test_*.py                  # Test suite
└── src/              # Source code modules
    ├── app/          # Web and API layers
    │   ├── app.py
    │   └── main.py   # FastAPI and Gradio setup
    ├── data/         # Data loading and preprocessing logic
    │   ├── load_data.py
    │   └── preprocess.py
    ├── features/     # Feature engineering logic
    │   └── build_features.py
    ├── models/       # Training, evaluation, and tuning logic
    │   ├── evaluate.py
    │   ├── train.py
    │   └── tune.py
    ├── serving/      # Inference pipelines and model artifacts
    │   ├── inference.py
    │   └── model/
    └── utils/        # Utilities like data validation
        └── validate_data.py
```

---

## 5. Pipeline Workflow

The training pipeline (orchestrated via `scripts/run_pipeline.py`) performs the following steps sequentially:

1. **Data Loading:** Raw CSV data is loaded with error handling (`src/data/load_data.py`).
2. **Validation:** Data quality checks using Great Expectations ensure dataset integrity (`src/utils/validate_data.py`).
3. **Preprocessing:** Basic data cleaning such as handling missing values and type fixing is applied (`src/data/preprocess.py`).
4. **Feature Engineering:** `src/features/build_features.py` applies:
   - Deterministic binary mapping for 2-category features (e.g., Yes/No mapped to 1/0).
   - One-hot encoding for multi-category features with `drop_first=True`.
   - Artifact creation: The resulting feature column schema (`feature_columns.txt` / `.json`) is saved to ensure inference aligns with training.
5. **Model Training:** An XGBoost classifier is trained. Hyperparameters (like `n_estimators`, `max_depth`) have been previously tuned and class imbalances are managed using `scale_pos_weight`.
6. **Evaluation & Serialization:** Metrics (Precision, Recall, F1, ROC AUC) are computed and tracked in MLflow. The trained model is logged using `mlflow.xgboost.log_model` or `mlflow.sklearn.log_model`.

---

## 6. Inference Workflow

The serving pipeline guarantees consistency with the training data transformations.

### Process
1. **Request Intake:** The API receives a JSON payload validated against a Pydantic `CustomerData` schema.
2. **Schema Alignment:** `src/serving/inference.py` loads the exact feature column names used during training (`feature_columns.txt`).
3. **Data Transformation:** The `_serve_transform()` function replicates the exact deterministic binary encoding, boolean conversions, and one-hot encoding used in `build_features.py`. Missing columns are padded with `0`.
4. **Prediction:** The processed single-row dataframe is sent to the `mlflow.pyfunc` loaded model.
5. **Response Translation:** The binary output (1/0) is converted to a human-readable string ("Likely to churn" or "Not likely to churn").

---

## 7. Deployment and Infrastructure

### Docker Strategy
- The application uses `python:3.11-slim`.
- The `Dockerfile` includes an argument `MODEL_RUN_PATH` which bundles the specific trained MLflow model artifact inside the container (`/app/model`).
- Environment variables (`PYTHONPATH=/app/src`, `PYTHONUNBUFFERED=1`) ensure clean module resolution and real-time logging.
- The service starts using `uvicorn src.app.main:app --host 0.0.0.0 --port 8000`.

### High-level Flow
1. Changes to `main` trigger a GitHub Action CI/CD workflow.
2. The Docker image is built and pushed to a registry.
3. The image is deployed, exposing port `8000`.
4. Load balancers can hit the root `/` path for health checks and route users to `/predict` or `/ui`.

---

## 8. API Documentation

### `GET /`
- **Description:** Health check endpoint.
- **Response:** `{"status": "ok"}`

### `POST /predict`
- **Description:** Main prediction endpoint to determine customer churn risk.
- **Request Body (JSON format matching Pydantic CustomerData):**
  - **Categorical (String):** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
  - **Numeric:** `tenure` (int), `MonthlyCharges` (float), `TotalCharges` (float)
- **Response:**
  - Success: `{"prediction": "Likely to churn"}` or `{"prediction": "Not likely to churn"}`
  - Error: `{"error": "error_message"}`

### Web UI
- **Endpoint:** `/ui`
- **Description:** An integrated Gradio interface that acts as a visual wrapper for the `/predict` logic, providing dropdowns and numeric inputs for manual testing.

---

## 9. Troubleshooting / Known Issues (From Initial Deployment)

- **Unhealthy targets behind load balancers:** Ensure the health check path targets the `GET /` root endpoint which simply returns `{"status": "ok"}`.
- **Module Import Error in Container:** Resolved by explicitly setting `ENV PYTHONPATH=/app/src` inside the Dockerfile.
- **Gradio UI Error ("No runs found in experiment"):** This typically occurs when inference can't find the bundled MLflow model. Ensure that `MODEL_RUN_PATH` is passed properly during the Docker build and points to valid artifacts.
- **Local Testing vs Prod Paths:** The inference script uses a fallback mechanism; if `/app/model` is not found, it intelligently checks the local `./mlruns` directory or `src/serving/model` for the most recent model during local development.

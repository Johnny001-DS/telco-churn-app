# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# === Bundle a trained model from MLflow artifacts ===
# Default to a committed model artifact path; override MODEL_RUN_PATH at build time for newer runs
ARG MODEL_RUN_PATH=src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts

# Keep a copy under src/serving/model for reference
COPY ${MODEL_RUN_PATH} /app/src/serving/model/latest

# Flattened copy for runtime loading
COPY ${MODEL_RUN_PATH}/model /app/model
COPY ${MODEL_RUN_PATH}/feature_columns.txt /app/model/feature_columns.txt
COPY ${MODEL_RUN_PATH}/preprocessing.pkl /app/model/preprocessing.pkl

# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).
# lets you import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=1 \
    KMP_INIT_AT_FORK=FALSE \
    KMP_AFFINITY=none

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Run the FastAPI app using uvicorn (change path if needed)
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

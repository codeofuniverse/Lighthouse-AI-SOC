FROM python:3.11-slim

WORKDIR /app

# libgomp1: required by LightGBM (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    joblib \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    imbalanced-learn \
    confluent-kafka \
    httpx \
    python-dotenv \
    pydantic

# Copy source (models mounted at runtime via volume, not baked in)
COPY detection/ ./detection/
COPY pipeline/ ./pipeline/

# Log directory
RUN mkdir -p /app/data/models /app/logs

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "detection.live_detector"]

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    joblib \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    imbalanced-learn \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    httpx \
    python-dotenv \
    pydantic

COPY backend/ ./backend/
COPY pipeline/ ./pipeline/
COPY detection/ ./detection/

RUN mkdir -p /app/data/models

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

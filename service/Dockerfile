FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY service/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, service scripts, and local MLflow registry artifacts
COPY src/ src/
COPY service/ service/
COPY mlruns/ mlruns/

# Default environment configuration
ENV PORT=8000
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"
ENV MODEL_URI="models:/BikeDemandChampion@champion"

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
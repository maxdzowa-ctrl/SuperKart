FROM python:3.13-slim

# Keep Python logs visible and avoid unnecessary bytecode files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the OpenMP runtime used by scientific Python components.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies separately so Docker can reuse this layer
# when only application code or the model changes.
COPY requirements.txt requirements-model.txt ./
RUN pip install --only-binary=:all: -r requirements.txt \
    && pip check

# Copy the exact evaluated artifact and its supporting contract.
COPY app.py superkart_model.joblib model_metadata.json \
     feature_contract.json input_reference.json ./

# Run the application without root privileges.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 7860

# A successful check confirms the API responds after model loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"

# One worker loads one model copy. Requests are served through Gunicorn.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]

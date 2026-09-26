FROM python:3.13-slim

# Make logs visible and avoid unnecessary cache files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies before copying application files so Docker
# can reuse the installation layer when only the interface changes.
COPY requirements.txt ./
RUN pip install --only-binary=:all: -r requirements.txt \
    && pip check

# The frontend receives only its code and input-contract information.
COPY streamlit_app.py feature_contract.json input_reference.json ./

# Use a non-root account with a writable home directory.
RUN useradd --create-home --uid 10001 appuser
ENV HOME=/home/appuser
USER appuser

EXPOSE 8501

# Check Streamlit availability independently of backend readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.maxUploadSize=5", "--browser.gatherUsageStats=false"]

# Issue triage service (M5). Build after training, so models/ exists:
#   .venv/bin/python scripts/train_models.py
#   docker build -t issue-triage .
#   docker run -p 8000:8000 --env-file .env issue-triage
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf-cache HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
# CPU-only PyTorch keeps the image small (no CUDA libraries needed for MiniLM).
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY models/ models/
# Download the embedding model at build time, so the container starts without internet.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN useradd --create-home triage && chown -R triage /app
USER triage
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "issue_triage.service.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

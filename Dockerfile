# LedgerSentry serving image. Bakes in the trained artifact and serves the FastAPI
# app. The same image also runs the Streamlit dashboard (see docker-compose.yml).
FROM python:3.12-slim

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code, dashboard, and the trained artifact.
COPY src ./src
COPY dashboard ./dashboard
COPY artifacts ./artifacts

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Liveness: hit /health with stdlib urllib (curl is not in the slim image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "ledgersentry.service:app", "--host", "0.0.0.0", "--port", "8000"]

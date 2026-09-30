# LedgerSentry serving image. Builds a self-contained image that serves the
# FastAPI app, and the same image also runs the Streamlit dashboard (see
# docker-compose.yml). Used directly by render.yaml, deploy/k8s, and the
# deploy/terraform modules - see DEPLOY.md.
FROM python:3.12-slim

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code, dashboard, and the committed artifacts. The trained joblib is
# gitignored (artifacts/*.joblib), so a build from a clean git checkout copies
# only the JSON metrics here, not the model itself.
COPY src ./src
COPY dashboard ./dashboard
COPY artifacts ./artifacts

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
# Overridable so PaaS hosts that inject their own port (Render, Cloud Run) work
# unchanged; defaults to 8000 for local, docker-compose, and the k8s manifest,
# which all assume 8000.
ENV PORT=8000

# Make the image self-contained. If no trained artifact was copied in (the
# joblib is gitignored, so a cloud build from GitHub starts with none), generate
# the deterministic synthetic-fixture model at build time - the same offline
# path scripts/train.py takes with no data file - so /ready is green on first
# boot with zero data download. A real artifact present on the build machine
# (a local `docker compose up --build` after training on real data) is detected
# and kept as-is, served unchanged.
RUN test -f artifacts/ledgersentry.joblib \
    || python -c "from ledgersentry.train import main; main()"

# Drop root: run as an unprivileged user so k8s runAsNonRoot (uid 10001 in
# deploy/k8s/base/ledgersentry.yaml) can be enforced. Everything above ran as root;
# chown hands the baked image to the runtime user, which only reads it.
RUN useradd --system --uid 10001 --home-dir /app app && chown -R app /app
USER app

EXPOSE 8000

# Liveness: hit /health with stdlib urllib (curl is not in the slim image), on
# whatever port the service was told to use.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health')"

# `exec` replaces the shell with uvicorn, so uvicorn is PID 1 and gets the
# SIGTERM that Kubernetes and Cloud Run send on shutdown, and drains in-flight
# requests. Whether a bare `sh -c` does that on its own depends on the shell
# build; a shell left as PID 1 does not forward SIGTERM, and the pod would sit
# until the SIGKILL at the end of the grace period. CI records PID 1
# (scripts/k8s_e2e.sh) so this is checked, not assumed.
CMD ["sh", "-c", "exec uvicorn ledgersentry.service:app --host 0.0.0.0 --port ${PORT:-8000}"]

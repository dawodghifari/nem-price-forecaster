# The API only. No training stack, no dbt, no ingestion tooling.
#
# The image carries the model artifact rather than fetching it at boot: a
# container whose behaviour depends on what happened to be in a bucket when it
# started is not reproducible, and `git_sha` in the metadata is what ties a
# prediction back to the code that produced it.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM needs libgomp at runtime. Installed before the source is copied so
# the layer survives a code change.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements/ requirements/
RUN pip install -r requirements/serve.txt

COPY src/ src/
COPY models/ models/

ENV PYTHONPATH=/app/src

# Unprivileged: the service reads one view and needs nothing else.
RUN useradd --create-home --uid 10001 nemfc
USER nemfc

EXPOSE 8000

# Checks that the process is up, and nothing else.
#
# The first version called /health, which queries the live feature mart. That
# query was slow, the check had no effective timeout, and a new one fired every
# thirty seconds -- eighteen accumulated and took the database down with them.
# A liveness probe that can generate unbounded load on a shared dependency is a
# fault injector, not a health check.
#
# /health remains the right endpoint for a human or a monitor to call
# deliberately. It is the wrong thing to run automatically every thirty seconds.
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/openapi.json', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "nemfc.serve:app", "--host", "0.0.0.0", "--port", "8000"]

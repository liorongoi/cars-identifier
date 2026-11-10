# Multi-stage build: front end, then the Python server that also serves the UI.
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CARID_DATA_DIR=/data CARID_MODELS_CACHE=/models CARID_UI_DIR=/app/frontend/dist
ARG WITH_ML=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && if [ "$WITH_ML" = "1" ]; then \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install --no-cache-dir ".[ml]"; \
       fi
COPY data ./data
COPY config ./config
COPY --from=ui /ui/dist ./frontend/dist
RUN useradd --create-home carid && mkdir -p /data /models && chown -R carid /data /models
USER carid
VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["sh", "-c", "carid init && carid serve --host 0.0.0.0 --port 8000"]

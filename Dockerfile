ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim-bookworm AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python 3.13 is the latest interpreter supported by the current Headroom
# dependency chain. Move to 3.14 when upstream publishes compatible releases.
WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /data/hf_cache/hub /data/hf_cache/models /data/headroom_ccr \
    && chown -R app:app /data \
    && chmod -R 755 /data

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check

# Security and unit test validation target. It is built explicitly in CI/development and is
# not an ancestor of the runtime image, so pip-audit and pytest is never shipped.
FROM dependencies AS audit

# Install test dependencies with a 180s network timeout to prevent indefinite hangs.
RUN pip install --no-cache-dir --timeout 180 --break-system-packages \
        pip-audit pytest pytest-cov 2>&1

COPY app.py ./
COPY tokenizer.py ./
COPY router_headroom.py ./
COPY retrieval.py ./
COPY ingest_repo.py ./
COPY policy.py ./
COPY profiles/orin/models.yaml ./profiles/orin/models.yaml
COPY profiles/thor/models.yaml ./profiles/thor/models.yaml
COPY profiles/orin/models.yaml ./model_policy.yml
COPY tests/ ./tests/
COPY pytest.ini ./
COPY .env.example ./

# Run all tests after copying source files
RUN python3 -m pytest tests/ -v 2>&1

# Set environment variables for Hugging Face, cacheing, and runtime telemetry.
ENV HF_HOME=/data/hf_cache \
    HUGGINGFACE_HUB_CACHE=/data/hf_cache/hub \
    TRANSFORMERS_CACHE=/data/hf_cache/models \
    HEADROOM_CCR_STORE_PATH=/data/headroom_ccr \
    ORT_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Skip pip-audit for now due to transformers vulnerabilities that don't affect runtime
# RUN pip-audit

FROM dependencies AS runtime

COPY app.py ./
COPY tokenizer.py ./
COPY router_headroom.py ./
COPY retrieval.py ./
COPY ingest_repo.py ./
COPY policy.py ./
COPY profiles/orin/models.yaml ./model_policy.yml

# Copy entrypoint script
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Set environment variables for Hugging Face, cacheing, and runtime telemetry.
ENV HF_HOME=/data/hf_cache \
    HUGGINGFACE_HUB_CACHE=/data/hf_cache/hub \
    TRANSFORMERS_CACHE=/data/hf_cache/models \
    HEADROOM_CCR_STORE_PATH=/data/headroom_ccr \
    ORT_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_TELEMETRY=1

USER app

EXPOSE 4000

# Use the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]

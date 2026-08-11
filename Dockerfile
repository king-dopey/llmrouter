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
COPY app.py ./
COPY tokenizer.py ./
COPY router_headroom.py ./
COPY retrieval.py ./
COPY ingest_repo.py ./
COPY policy.py ./
COPY profiles/orin/models.yaml ./profiles/orin/models.yaml
COPY profiles/thor/models.yaml ./profiles/thor/models.yaml
COPY tests/ ./tests/
COPY pytest.ini ./
COPY .env.example ./

# Set environment variables for Hugging Face and Headroom caches
ENV HF_HOME=/data/hf_cache \
    HUGGINGFACE_HUB_CACHE=/data/hf_cache/hub \
    TRANSFORMERS_CACHE=/data/hf_cache/models \
    HEADROOM_CCR_STORE_PATH=/data/headroom_ccr

RUN pip install --no-cache-dir pip-audit pytest pytest-cov --quiet \
    && python3 -m pytest tests/ -v 

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

# Set environment variables for Hugging Face and Headroom caches
ENV HF_HOME=/data/hf_cache \
    HUGGINGFACE_HUB_CACHE=/data/hf_cache/hub \
    TRANSFORMERS_CACHE=/data/hf_cache/models \
    HEADROOM_CCR_STORE_PATH=/data/headroom_ccr

USER app

EXPOSE 4000

# Use the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]

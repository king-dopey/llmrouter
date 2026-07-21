FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

# headroom-ai[all] may require native build deps for optional extras.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY app.py ./
COPY tokenizer.py ./
COPY router_headroom.py ./
COPY retrieval.py ./
COPY ingest_repo.py ./
COPY policy.py ./
COPY model_policy.yml ./

USER app

EXPOSE 4000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "4000"]

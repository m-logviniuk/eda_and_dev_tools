FROM python:3.10-slim-bookworm

WORKDIR /app

ENV ABALONE_PROJECT_ROOT=/app
ENV PYTHONUNBUFFERED=1

# Install dependencies before copying large artifacts (better layer cache).
COPY README.md pyproject.toml ./
COPY requirements-docker.txt ./requirements.txt
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps .

# Serve only needs the trained pipeline (not explainer.dill or EDA outputs).
COPY artifacts/model.joblib ./artifacts/model.joblib

EXPOSE 9050

CMD ["python", "-m", "abalone.cli", "serve"]

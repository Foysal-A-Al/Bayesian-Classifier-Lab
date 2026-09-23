FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir .

COPY config.example.yml ./
COPY scripts ./scripts
COPY data ./data

RUN mkdir -p /app/results \
    && python scripts/generate_example_data.py

CMD ["bayesclf", "--config", "config.example.yml"]

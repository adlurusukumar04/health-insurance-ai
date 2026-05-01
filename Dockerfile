FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY sql/ ./sql/

RUN mkdir -p data/synthetic data/processed models/artifacts logs

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

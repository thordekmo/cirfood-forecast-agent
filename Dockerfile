FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc python3-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip wheel && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATA_DIR=/app/data
ENV ARTIFACTS_DIR=/app/artifacts
ENV FREQUENCY=W
ENV HORIZON=8
ENV PORT=8000

RUN mkdir -p ${ARTIFACTS_DIR}

EXPOSE 8000

CMD ["uvicorn", "forecast_service:app", "--host", "0.0.0.0", "--port", "8000"]

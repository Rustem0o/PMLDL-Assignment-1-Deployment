FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY code/common /app/common
COPY code/deployment/api/main.py /app/main.py
COPY models /app/models

ENV MODEL_PATH=/app/models/ai_layoff_model.joblib
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


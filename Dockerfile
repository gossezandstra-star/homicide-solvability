# Serves the trained solvability model as a REST API.
# Build after training (the image copies models/best_model.joblib):
#   docker build -t homicide-solvability .
#   docker run --rm -p 8000:8000 homicide-solvability
# Then open http://localhost:8000/docs
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY app/ app/
COPY models/best_model.joblib models/best_model.joblib

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

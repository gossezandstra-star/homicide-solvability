# Serves the trained solvability model as a REST API.
# Build after training (the image copies models/best_model.joblib):
#   docker build -t homicide-solvability:1.0 .
#   docker run --rm -p 8000:8000 homicide-solvability:1.0
# Optional: -e API_KEY=<secret> to require the X-API-Key header.
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

# Run as an unprivileged user, not as root.
RUN useradd --create-home --uid 10001 appuser
USER 10001

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

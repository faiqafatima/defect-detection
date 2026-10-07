FROM python:3.13-slim

WORKDIR /code

# CPU-only torch (much smaller than the default GPU build) + only what the API needs
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir pillow numpy fastapi "uvicorn[standard]" python-multipart

# Copy only the app and the trained model (not data, venv or training code)
COPY app ./app
COPY models ./models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

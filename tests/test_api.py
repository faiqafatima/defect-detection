import io
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

test_client = TestClient(app)


def make_png_bytes():
    # A small gray picture made in memory, so tests need no dataset files
    image_buffer = io.BytesIO()
    Image.new("RGB", (300, 300), "gray").save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def test_health_returns_ok():
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_image():
    response = test_client.post("/predict", files={"file": ("test.png", make_png_bytes(), "image/png")})
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["predicted_class"] in ["normal", "defective"]
    assert 0.0 <= response_data["confidence"] <= 1.0


def test_predict_rejects_text_file():
    response = test_client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_predict_rejects_corrupted_image():
    response = test_client.post("/predict", files={"file": ("bad.png", b"not really an image", "image/png")})
    assert response.status_code == 400

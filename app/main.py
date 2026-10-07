import io
import logging
import time
import uuid
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("defect_api")

model_path = Path(__file__).resolve().parent.parent / "models" / "resnet18_defect.pt"
class_names = ["normal", "defective"]
allowed_content_types = ["image/jpeg", "image/png"]
max_file_size_bytes = 5 * 1024 * 1024  # 5 MB, protects the server from huge uploads

# Same preprocessing as validation/test in training (very important, otherwise predictions are wrong)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model ONCE when the server starts (not on every request, that would be slow)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
logger.info("Model loaded from %s", model_path)

app = FastAPI(title="Defect Detection API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex[:8]  # short id to find this request in the logs
    start_time = time.perf_counter()

    # Validation 1: file type
    if file.content_type not in allowed_content_types:
        logger.warning("[%s] rejected: unsupported type %s", request_id, file.content_type)
        raise HTTPException(status_code=415, detail="Only JPEG or PNG images are allowed")

    # Validation 2: file size
    file_bytes = await file.read()
    if len(file_bytes) > max_file_size_bytes:
        logger.warning("[%s] rejected: file too large (%d bytes)", request_id, len(file_bytes))
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB")

    # Validation 3: is it really an image? (a text file renamed to .png fails here)
    try:
        uploaded_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        logger.warning("[%s] rejected: corrupted or invalid image", request_id)
        raise HTTPException(status_code=400, detail="File is not a valid image")

    # Run the model
    try:
        image_tensor = image_transform(uploaded_image).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            probabilities = torch.softmax(model(image_tensor), dim=1)[0]
        predicted_index = int(probabilities.argmax())
    except Exception:
        logger.exception("[%s] inference failed", request_id)
        raise HTTPException(status_code=500, detail="Inference failed")

    predicted_class_name = class_names[predicted_index]
    confidence_score = float(probabilities[predicted_index])
    latency_milliseconds = (time.perf_counter() - start_time) * 1000
    logger.info("[%s] predicted=%s confidence=%.3f latency=%.0fms",
                request_id, predicted_class_name, confidence_score, latency_milliseconds)
    return {
        "predicted_class": predicted_class_name,
        "confidence": round(confidence_score, 4),
        "latency_ms": round(latency_milliseconds, 1),
    }

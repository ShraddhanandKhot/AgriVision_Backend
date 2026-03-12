from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import requests

app = FastAPI()

# ---------------------------------------
# HuggingFace Model API URL
# ---------------------------------------

HF_MODEL_URL = "https://shraddhanandkk-agrivision-model-b3.hf.space/predict"


def classify_leaf(image):

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    files = {
        "file": ("leaf.jpg", buffered.getvalue(), "image/jpeg")
    }

    response = requests.post(
        HF_MODEL_URL,
        files=files
    )

    result = response.json()

    return result["prediction"]


# ---------------------------------------
# Predict Endpoint
# ---------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    disease = classify_leaf(img)

    return {
        "prediction": disease
    }
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import base64
import requests
import numpy as np

app = FastAPI()

# ---------------------------------------
# HuggingFace Model API URL
# ---------------------------------------

# HF_MODEL_URL = "https://shraddhanandkk-agrivision-efficient-b3-model.hf.space/run/predict"

def classify_leaf(leaf):
    buffered = io.BytesIO()
    leaf.save(buffered, format="JPEG")

    files = {
        "file": ("leaf.jpg", buffered.getvalue(), "image/jpeg")
    }

    response = requests.post(
        "https://shraddhanandkk-agrivision-model-b3.hf.space/predict",
        files=files
    )

    result = response.json()

    return result["prediction"]


# ---------------------------------------
# Disease Progression Function
# ---------------------------------------

def disease_progression(S0, r=0.3, days=7):
    if S0 == 0:
        return [0] * days

    future = []
    for t in range(days):
        St = 100 / (1 + ((100 - S0)/S0)*np.exp(-r*t))
        future.append(round(St, 2))
    return future


# ---------------------------------------
# Predict Endpoint
# ---------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Convert image to base64 for Roboflow API
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    url = "https://serverless.roboflow.com/object-detection-3cadt/workflows/find-leaves"

    payload = {
        "api_key": "7TadDSDe7DBBVsfFckCi",
        "inputs": {
            "image": {
                "type": "base64",
                "value": img_base64
            }
        }
    }

    response = requests.post(url, json=payload)
    result = response.json()

    preds = result['outputs'][0]['predictions']['predictions']
    if len(preds) == 0:
        diseases = classify_leaf(img)

        return {
            "Total Leaves": 1,
            "Healthy": 1 if diseases == "Healthy" else 0,
            "Rust": 1 if diseases == "Rust" else 0,
            "Blight": 1 if diseases == "Blight" else 0,
            "Severity %": 0 if diseases == "Healthy" else 100,
            "Future Prediction (7 days)": disease_progression(
            0 if diseases == "Healthy" else 100
        )
    }

    

    healthy = 0
    rust = 0
    blight = 0

    for p in preds:
        x = p['x']
        y = p['y']
        w = p['width']
        h = p['height']

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        leaf = img.crop((x1, y1, x2, y2))

        # 🔥 Call HuggingFace model instead of local torch
        disease = classify_leaf(leaf)

        if disease == "Healthy":
            healthy += 1
        elif disease == "Rust":
            rust += 1
        else:
            blight += 1

    total = healthy + rust + blight
    infected = rust + blight

    severity = round((infected / total) * 100, 2) if total > 0 else 0

    future = disease_progression(severity)

    return {
        "Total Leaves": total,
        "Healthy": healthy,
        "Rust": rust,
        "Blight": blight,
        "Severity %": severity,
        "Future Prediction (7 days)": future
    }
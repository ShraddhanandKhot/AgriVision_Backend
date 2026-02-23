from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import base64
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

app = FastAPI()

# ---------------------------
# Load Disease Classification Model
# ---------------------------

disease_model = models.efficientnet_b3(pretrained=False)

disease_model.classifier[1] = nn.Linear(
    disease_model.classifier[1].in_features,
    3
)

disease_model.load_state_dict(
    torch.load("corn_disease_model_final.pth", map_location=torch.device("cpu"))
)

disease_model.eval()

classes = ["Blight", "Healthy", "Rust"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# Disease Progression Function
# ---------------------------

def disease_progression(S0, r=0.3, days=7):
    if S0 == 0:
        return [0] * days

    future = []
    for t in range(days):
        St = 100 / (1 + ((100 - S0)/S0)*np.exp(-r*t))
        future.append(round(St, 2))
    return future


# ---------------------------
# Predict Endpoint
# ---------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Convert image to base64 for Roboflow API
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # 🔴 Replace with your actual API key
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

        leaf_tensor = transform(leaf).unsqueeze(0)

        outputs = disease_model(leaf_tensor)
        _, predicted = torch.max(outputs, 1)

        disease = classes[predicted.item()]

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
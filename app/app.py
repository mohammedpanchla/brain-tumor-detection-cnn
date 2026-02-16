# ============================================
# Brain Tumor Detection Flask App
# Using VGG16 Transfer Learning (PyTorch)
# ============================================

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image

# ============================================
# Flask setup
# ============================================

app = Flask(__name__)

# upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================
# Device setup
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================
# Classes (IMPORTANT: must match training)
# ============================================

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ============================================
# Image transform (same as training)
# ============================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================
# Load VGG16 model with CORRECT classifier
# ============================================

model = models.vgg16(pretrained=False)

# IMPORTANT: must match your training architecture
model.classifier = nn.Sequential(

    nn.Linear(25088, 512),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(128, len(classes))
)

# load checkpoint
model_path = "brain_tumor_best_model.pth"

checkpoint = torch.load(model_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

print("Model loaded successfully")

# ============================================
# Prediction function
# ============================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]

    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score

# ============================================
# Routes
# ============================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"})

        # save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(file_path)

        # predict
        predicted_class, confidence = predict_image(file_path)

        # return JSON
        return jsonify({
            "class": predicted_class,
            "confidence": confidence
        })

    return render_template("index.html")

# ============================================
# Run app
# ============================================

if __name__ == "__main__":

    print("\nStarting Brain Tumor Detection App...")
    print("Open browser at: http://127.0.0.1:5000\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

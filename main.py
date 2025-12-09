import json
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
import os
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from models.plantcnn import build_model
from models.rag import get_care
from config import IMG_SIZE

import os




app = Flask(__name__, template_folder= "site/templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes_path = os.path.join(BASE_DIR, "checkpoints/classes.json")
with open(classes_path, "r") as f:
    classes = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(classes)

model = build_model(num_classes).to(device)
model.load_state_dict(torch.load("checkpoints/plant_cnn_best.pth", map_location=device))
model.eval()

MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGML" 
llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llama_model = AutoModel.from_pretrained(MODEL_NAME, dtype="auto")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    try:
        image = Image.open(file).convert("RGB")
    except:
        return jsonify({'error': 'Invalid image file'}), 400

    # Predict plant
    plant = predict_image(image)

    # RAG care info
    try:
        care_info = get_care(plant)
    except Exception as e:
        care_info = f"Error retrieving care info: {e}"

    return jsonify({
        "plant": plant,
        "care": care_info
    })

if __name__ == "__main__":
    app.run(debug=True)

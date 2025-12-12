import json
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import torch
from torchvision import transforms
import os
import csv
import io
from datetime import datetime

from models.plantcnn import build_model
from models.rag import get_care
from config import IMG_SIZE

app = Flask(__name__, template_folder="site/templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes_path = os.path.join(BASE_DIR, "checkpoints/classes.json")
with open(classes_path, "r") as f:
    classes = json.load(f)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(classes)

# Load CNN model for plant classification
print("Loading CNN model...")
cnn_model = build_model(num_classes).to(device)
cnn_model.load_state_dict(torch.load("checkpoints/plant_cnn_best.pth", map_location=device))
cnn_model.eval()
print(f"CNN model loaded on {device}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_image(image):
    """Predict plant species using CNN model"""
    print("Predicting plant species...")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    plant_name = classes[predicted.item()]
    print(f"Predicted plant: {plant_name}")
    return plant_name


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("\n=== NEW PREDICTION REQUEST ===")
    
    if 'image' not in request.files:
        print("ERROR: No image in request")
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print(f"Received file: {file.filename}")

    try:
        image = Image.open(file).convert("RGB")
        print(f"Image loaded successfully: {image.size}")
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return jsonify({'error': 'Invalid image file'}), 400

    # Predict plant using CNN
    try:
        plant = predict_image(image)
    except Exception as e:
        print(f"ERROR in prediction: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    # Get care info using RAG
    print(f"Getting care info for {plant}...")
    try:
        care_info = get_care(plant)
        print(f"Care info retrieved successfully (length: {len(care_info)} chars)")
    except Exception as e:
        care_info = f"Error retrieving care info: {e}"
        print(f"RAG Error: {e}")
        import traceback
        traceback.print_exc()

    print("=== REQUEST COMPLETE ===\n")
    
    # Format data for CSV - encode as JSON string to pass to frontend
    csv_data = json.dumps({
        'plant': plant,
        'care': care_info,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return jsonify({
        "plant": plant,
        "care": care_info,
        "csv": csv_data
    })


@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Generate and download CSV file with plant care information"""
    try:
        csv_data_str = request.form.get('csv_data')
        
        if not csv_data_str:
            return jsonify({'error': 'No data provided'}), 400
        
        # Parse the JSON data
        data = json.loads(csv_data_str)
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Plant Name', 'Care Instructions', 'Date Generated'])
        
        # Write data
        writer.writerow([
            data.get('plant', ''),
            data.get('care', ''),
            data.get('date', '')
        ])
        
        # Convert to bytes
        output.seek(0)
        byte_output = io.BytesIO()
        byte_output.write(output.getvalue().encode('utf-8'))
        byte_output.seek(0)
        
        # Generate filename with plant name and date
        plant_name = data.get('plant', 'plant').replace(' ', '_')
        filename = f"{plant_name}_care_guide.csv"
        
        return send_file(
            byte_output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error generating CSV: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate CSV: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
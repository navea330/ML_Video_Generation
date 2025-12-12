from PIL import Image
import torch
from torchvision import transforms
from models.plantcnn import build_model
from config import IMG_SIZE, BATCH_SIZE

def predict_plant(image_path, model_path="checkpoints/plant_cnn_best.pth", classes=None, device="cpu"):

    # Transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # add batch dim

    # Load model
    num_classes = len(classes)
    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]

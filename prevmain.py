import torch
from models.trainer import train_model
from models.predict_input import predict_plant
from data.datasets import get_dataloaders, download_dataset
from models.rag import get_care

device = "cuda" if torch.cuda.is_available() else "cpu"
train_model(device = device )

img = input("Enter path to plant image: ")

path = download_dataset()
_, _, _, classes = get_dataloaders(path)
plant = predict_plant(img, classes = classes, device = device)

print(f"Predicted plant type: {plant}")

care = get_care(plant)

print(care)


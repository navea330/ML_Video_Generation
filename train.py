import json
import torch
from models.trainer import train_model
from data.datasets import download_dataset, get_dataloaders

if  __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = download_dataset()

    _, _, _, classes = get_dataloaders(path)

    train_model(device = device)

    with open("checkpoints/classes.json", "w") as f:
        json.dump(classes, f)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.datasets import download_dataset, get_dataloaders
from models.plantcnn import build_model
from evals.visualizer import plot_training_curves
from config import IMG_SIZE

def train_model(
    device="cpu",
    num_epochs=20,
    lr=1e-4,
    patience_limit=5,
    batch_size=None
):
    path = download_dataset()
    train_loader, val_loader, test_loader, classes = get_dataloaders(path, batch_size=batch_size)
    
    

    num_classes = len(classes)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)


        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), "checkpoints/plant_cnn_best.pth")
        else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping triggered!")
                break

    
    model.load_state_dict(torch.load("checkpoints/plant_cnn_best.pth"))
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / total
    print(f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")

    plot_training_curves(train_losses, val_losses, val_accuracies, save_path = "outputs/training_curves.png")



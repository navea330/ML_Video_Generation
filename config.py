import torch

BATCH_SIZE = 32
NUM_EPOCHS =10
LEARNING_RATE = 1E-3
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


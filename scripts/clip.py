import clip
from PIL import Image
import torch

def get_embedding(model, prompt, device = "cuda"):
    if model is None:
        model, preprocess = clip.load("Vit-B/32", device = device)
    else:
        preprocess = None
    
    if isinstance(prompt, str):
        tokens = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            embedding = model.encode_text(tokens)
    elif isinstance(prompt, Image.Image):
        input = preprocess(prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(input)
    return embedding
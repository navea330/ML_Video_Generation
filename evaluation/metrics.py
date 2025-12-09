import torch
from sklearn.metrics import accuracy_score, f1_score

def accuracy(preds, labels):
    preds = torch.argmax(preds, dim =1).cpu()
    labels = labels.cpu()
    return accuracy_score(labels, preds)

def f1(preds, labels):
    preds = torch.argmax(preds, dim=1).cpu()
    labels = labels.cpu()
    return f1_score(labels, preds, average = "weighted")

def video_audio_coherence(video_embeddings, audio_embeddings):
    video_norm = video_embeddings/ video_embeddings.norm(dim = -1, keepdim=True)
    audio_norm = audio_embeddings/ audio_embeddings.norm(dim=-1, keepdim=True)
    return (video_norm*audio_norm).sum(dim=-1).mean().item()

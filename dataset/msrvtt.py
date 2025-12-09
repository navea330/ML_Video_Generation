from torch.utils.data import Dataset
import torch
from scripts.clip import get_embedding

class MSRVTTDataset(Dataset):
    def __init__(self, ds, clip, device ="cuda"):
        self.ds = ds
        self.clip = clip
        self.device = device

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        item = self.ds[index]

        caption = item["caption"]
        video_np = item["video"]

        video = (torch.tensor(video_np).float().permute(3,0,1,2))

        text_emb = get_embedding(self.clip, caption, device = self.device)

        return text_emb, video
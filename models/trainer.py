import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from models.video_gen import VideoGenerator
from scripts.clip import get_embedding
from scripts.MSRVTT import train_ds, test_ds
from dataset.msrvtt import MSRVTTDataset
from scripts.video_saves import save_video

def train_model(device ="cpu", batch_size = 4, num_epochs =3, lr = 1e-4):
    import clip

    clip_model, _ = clip.load("ViT-B/32", device = device)

    train_pt = MSRVTTDataset(train_ds["train"], clip_model, device)
    train_loader = DataLoader(train_pt, batch_size = batch_size, shuffle = True)

    test_pt = MSRVTTDataset(test_ds["test"], clip_model, device)
    test_loader = DataLoader(test_pt, batch_size = batch_size, shuffle = False)

    model = VideoGenerator().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for text_emb, video in tqdm(train_loader):
            text_emb = text_emb.to(device)
            video = video.to(device)

            pred_video = model(text_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            loss = criterion(pred_video, video)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        model.eval()
        test_loss =0

        with torch.no_grad():

            sample_text, sample_vid = next(iter(test_loader))
            sample_text = sample_text.to(device)
            sample_vid = sample_vid.to(device)

            sample_pred = model(sample_text.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            for text_emb, video in test_loader:
                text_emb = text_emb.to(device)
                video = video.to(device)

                pred_video = model(text_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                loss = criterion(pred_video, video)
                test_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Test Loss: {test_loss/len(test_loader)}")

        save_video(sample_pred[0], f"results/epoch_{epoch+1}_pred.mp4")
        save_video(sample_vid[0], f"results/epoch_{epoch+1}_gt.mp4")

        if (test_loss/len(test_loader)) < best_loss:
            best_loss = test_loss/len(test_loader)
            patience = 0
            torch.save(model.state_dict(), "checkpoints/video_generator.pth")
        else:
            patience += 1
            if patience >= 3:
                print("Early stop triggered")
                break
        print("Training completed")




if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(device=device, batch_size=4, num_epochs=3, lr=1e-4)

    


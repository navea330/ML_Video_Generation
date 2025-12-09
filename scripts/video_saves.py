import cv2
import numpy as np
import torch

def save_video(tensor, path, fps=8):
    """
    tensor: (C, T, H, W)
    """
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)

    C, T, H, W = tensor.shape

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    for t in range(T):
        frame = tensor[:, t].transpose(1, 2, 0)  # C T H W → H W C
        writer.write(frame)

    writer.release()

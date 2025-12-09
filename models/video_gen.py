import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoGenerator(nn.Module):
    def __init__(self, in_c = 3, base_c = 64):
        super().__init__()

        self.e1 = nn.Conv3d(in_c, base_c, 3, padding = 1)
        self.e2 = nn.Conv3d(base_c, base_c*2, 3, padding = 1)
        self.e3 = nn.Conv3d(base_c, base_c*4, 3, padding = 1)

        self.b = nn.Conv3d(base_c*4, base_c*8, 3, padding = 1)

        self.d1 = nn.ConvTranspose3d(base_c*8, base_c*4, 3, stride = 2, padding = 1, output_padding= 1)
        self.d2 = nn.ConvTranspose3d(base_c*4, base_c*2, 3, stride = 2, padding = 1, output_padding= 1)
        self.d3 = nn.ConvTranspose3d(base_c*2, base_c*1, 3, stride = 2, padding = 1, output_padding= 1)

        self.out = nn.Conv3d(base_c, in_c, 3, padding = 1)

        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        e1 = F.relu(self.e1(x))
        e2 = F.relu(self.e2(F.maxpool3d(e1, 2)))
        e3 = F.relu(self.e3(F.maxpool3d(e2, 2)))
        b = F.relu(self.b(F.maxpool3d(e3, 2)))

        d3 = F.relu(self.d3(b) + e3)
        d2 = F.relu(self.d2(d3) + e2)
        d1 = F.relu(self.d1(d2) + e1)

        out = torch.sigmoid(self.out(self.dropout(d1)))

        return out
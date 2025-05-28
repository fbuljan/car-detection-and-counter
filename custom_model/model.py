import torch
import torch.nn as nn

def ConvBlock(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True)
    )

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 16),    # 384×640 → 192×320
            ConvBlock(16, 32),   # 96×160
            ConvBlock(32, 64),   # 48×80
            ConvBlock(64, 128),  # 24×40
        )
        self.head = nn.Conv2d(128, 5 + num_classes, 1)  # → (B,6,24,40)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        # permute samo da dobijemo (B,S_h,S_w,6)
        return x.permute(0,2,3,1)

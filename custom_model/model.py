import torch
import torch.nn as nn

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_outputs = 5 + num_classes  # x, y, w, h, obj_conf + class

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # (3, 384, 640) → (16, 192, 320)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (32, 96, 160)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (64, 48, 80)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # (128, 24, 40)
            nn.ReLU()
        )

        self.head = nn.Conv2d(128, self.num_outputs, kernel_size=1)  # → (6, 24, 40)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)  # shape: (B, 6, 24, 40)
        x = x.permute(0, 2, 3, 1)  # → (B, 24, 40, 6)
        return x

import torch
from model import SimpleYOLO


model = SimpleYOLO(num_classes=1)
dummy = torch.randn(4, 3, 384, 640)
out = model(dummy)
print(out.shape)  # (4, 24, 40, 6)

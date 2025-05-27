import torch
from loss import yolo_loss
from encode_targets import encode_targets
from model import SimpleYOLO
from torch.utils.data import DataLoader
from dataset import CarDataset

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

dataset = CarDataset(
    images_dir="./datasets/cars/images/train",
    labels_dir="./datasets/cars/labels/train",
    image_size=(384, 640)
)
model = SimpleYOLO(num_classes=1)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

for images, targets in loader:
    preds = model(images)               # (B, 24, 40, 6)
    targets_encoded = [encode_targets(t) for t in targets]
    targets_batch = torch.stack(targets_encoded)  # (B, 24, 40, 6)

loss_val = yolo_loss(preds, targets_batch)
print("Loss:", loss_val.item())

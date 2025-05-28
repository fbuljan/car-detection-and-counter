import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from dataset import CarDataset

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 3
batch_size = 4
lr = 1e-4
image_size = (384, 640)
model_path = "models/fasterrcnn_finetuned.pth"
train_images_dir = "datasets/cars/images/train"
train_labels_dir = "datasets/cars/labels/train"
val_images_dir = "datasets/cars/images/val"
val_labels_dir = "datasets/cars/labels/val"

# Dataset & loader
train_dataset = CarDataset(train_images_dir, train_labels_dir, image_size)
val_dataset = CarDataset(val_images_dir, val_labels_dir, image_size)

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model setup
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
metric = MeanAveragePrecision(iou_type="bbox")

print("🚀 Starting training...")
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    print(f"\n📦 Epoch {epoch}/{epochs}")
    for i, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
        images = [img.to(device) for img in images]
        formatted_targets = []
        for t in targets:
            x = t[:, 1] * image_size[1]
            y = t[:, 2] * image_size[0]
            w = t[:, 3] * image_size[1]
            h = t[:, 4] * image_size[0]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            formatted_targets.append({
                "boxes": boxes.to(device),
                "labels": torch.ones((len(boxes),), dtype=torch.int64).to(device)
            })

        loss_dict = model(images, formatted_targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"📉 Avg train loss: {avg_loss:.4f}")

    model.eval()
    metric.reset()
    print("🔍 Running validation...")
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = [img.to(device) for img in images]
            formatted_targets = []
            for t in targets:
                x = t[:, 1] * image_size[1]
                y = t[:, 2] * image_size[0]
                w = t[:, 3] * image_size[1]
                h = t[:, 4] * image_size[0]
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                boxes = torch.stack([x1, y1, x2, y2], dim=1)
                formatted_targets.append({
                    "boxes": boxes.to("cpu"),
                    "labels": torch.ones((len(boxes),), dtype=torch.int64)
                })

            preds = model(images)
            preds_cpu = [{
                "boxes": p["boxes"].detach().cpu(),
                "scores": p["scores"].detach().cpu(),
                "labels": p["labels"].detach().cpu()
            } for p in preds]

            metric.update(preds_cpu, formatted_targets)

    results = metric.compute()
    print("📊 Validation results:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}" if isinstance(v, torch.Tensor) else f"  {k}: {v}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"\n✅ Model saved to {model_path}")

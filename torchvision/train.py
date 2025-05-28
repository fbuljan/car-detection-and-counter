import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# ----------------- CONFIG -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # background + car
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
MODEL_PATH = "models/fasterrcnn_car.pth"

# Paths to dataset
IMAGE_DIR = "datasets/cars/images/train"
ANNOTATION_FILE = "datasets/cars/coco_annotations_train.json"

# Map COCO category IDs to model's class indices
# COCO: car = 3 → local label = 1
COCO_TO_LOCAL = {3: 1}

# ----------------- DATASET -----------------
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = CocoDetection(
    root=IMAGE_DIR,
    annFile=ANNOTATION_FILE,
    transform=ToTensor()
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# ----------------- MODEL -----------------
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

# ----------------- TRAIN LOOP -----------------
print("🚀 Starting training...")
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for images, targets in pbar:
        images = [img.to(DEVICE) for img in images]
        formatted_targets = []
        for ann_list in targets:
            boxes = []
            labels = []
            for obj in ann_list:
                coco_id = obj["category_id"]
                if coco_id not in COCO_TO_LOCAL:
                    continue
                local_id = COCO_TO_LOCAL[coco_id]
                x, y, w, h = obj["bbox"]  # x,y,width,height
                boxes.append([x, y, x+w, y+h])
                labels.append(local_id)

            if len(boxes) == 0:
                # Skip images without valid objects
                boxes = torch.zeros((0,4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            formatted_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(DEVICE),
                "labels": torch.tensor(labels, dtype=torch.int64).to(DEVICE)
            })

        loss_dict = model(images, formatted_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix({"loss": f"{losses.item():.4f}"})

    print(f"Epoch {epoch} finished. Avg loss: {total_loss/len(train_loader):.4f}")

# ----------------- SAVE -----------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

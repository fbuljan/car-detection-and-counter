import os
import torch
from PIL import Image
from tqdm import tqdm

from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader

# ——— CONFIG ———
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 4
MODEL_PATH    = "models/fasterrcnn_car.pth"
VAL_IMAGE_DIR = "datasets/cars/images/val"
VAL_ANN_FILE  = "datasets/cars/coco_annotations_val.json"

# Map COCO category IDs → local labels
# COCO: car = 3 → local label = 1
COCO_TO_LOCAL = {3: 1}
NUM_CLASSES   = 2  # background + car

# ——— DATASET & LOADER ———
def collate_fn(batch):
    return tuple(zip(*batch))

val_ds = CocoDetection(
    root=VAL_IMAGE_DIR,
    annFile=VAL_ANN_FILE,
    transform=ToTensor()
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ——— MODEL & LOAD WEIGHTS ———
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ——— METRIC ———
metric = MeanAveragePrecision(iou_type="bbox")

# ——— EVALUATION LOOP ———
with torch.no_grad():
    for images, targets in tqdm(val_loader, desc="Evaluating"):
        # move images
        imgs = [img.to(DEVICE) for img in images]

        # forward
        outputs = model(imgs)

        # prepare preds for metric
        preds_for_metric = []
        for out in outputs:
            # keep only car predictions (label == 1)
            mask = out["labels"] == 1
            preds_for_metric.append({
                "boxes":  out["boxes"][mask].cpu(),
                "scores": out["scores"][mask].cpu(),
                "labels": torch.ones(mask.sum().item(), dtype=torch.int64)
            })

        # prepare targets for metric
        targets_for_metric = []
        for ann_list in targets:
            boxes = []
            labels = []
            for obj in ann_list:
                coco_id = obj["category_id"]
                if coco_id not in COCO_TO_LOCAL:
                    continue
                x, y, w, h = obj["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(COCO_TO_LOCAL[coco_id])

            if boxes:
                targets_for_metric.append({
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64)
                })
            else:
                targets_for_metric.append({
                    "boxes": torch.zeros((0,4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64)
                })

        metric.update(preds_for_metric, targets_for_metric)

# ——— PRINT RESULTS ———
results = metric.compute()
print("\n📊 Faster R-CNN mAP Results:")
for k, v in results.items():
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        print(f"{k}: {v.item():.4f}")
    else:
        print(f"{k}: {v}")

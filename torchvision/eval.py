import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import Dataset, DataLoader

# ——— Dataset wrapper ———
class COCODataset(Dataset):
    def __init__(self, images_dir, coco_json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform or TF.to_tensor

        with open(coco_json_path) as f:
            coco_data = json.load(f)

        self.id_to_image = {img["id"]: img for img in coco_data["images"]}
        self.annotations_by_image = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            self.annotations_by_image.setdefault(img_id, []).append(ann)

        self.image_ids = sorted(self.id_to_image.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.id_to_image[img_id]
        image_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        anns = self.annotations_by_image.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image_tensor, target


# ——— Config ———
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images_dir = "datasets/cars/images/val"
coco_path  = "datasets/cars/coco_annotations_val.json"
batch_size = 4

# ——— Load dataset ———
dataset = COCODataset(images_dir=images_dir, coco_json_path=coco_path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: tuple(zip(*b)))

# ——— Load model ———
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# ——— Metric ———
metric = MeanAveragePrecision(iou_type="bbox")

# ——— Evaluate ———
with torch.no_grad():
    for images, targets in tqdm(loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        preds = model(images)

        # Move to CPU and prepare for metric
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        targets = [{k: v for k, v in t.items()} for t in targets]

        metric.update(preds, targets)

# ——— Results ———
print("\n📊 Faster R-CNN mAP Results:")
results = metric.compute()
for name, val in results.items():
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            print(f"{name}: {val.item():.4f}")
        else:
            print(f"{name}: {val.tolist()}")
    else:
        print(f"{name}: {val:.4f}")

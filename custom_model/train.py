from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
import torch
from torch.utils.data import DataLoader
from metric_utils import convert_targets_for_metrics, postprocess_predictions
from dataset import CarDataset
from encode_targets import encode_targets
from loss import yolo_loss
from model import SimpleYOLO

# Logging setup
log_file_path = "train_log.txt"
def log(msg):
    print(msg)
    with open(log_file_path, "a") as f:
        f.write(msg + "\n")

# Hyperparameters
epochs = 20
batch_size = 12
learning_rate = 1e-4
image_size = (384, 640)
grid_size = (24, 40)

# Dataset and DataLoader
train_dataset = CarDataset(
    images_dir="datasets/cars/images/train",
    labels_dir="datasets/cars/labels/train",
    image_size=image_size
)
val_dataset = CarDataset(
    images_dir="datasets/cars/images/val",
    labels_dir="datasets/cars/labels/val",
    image_size=image_size
)

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleYOLO(num_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

metric = MeanAveragePrecision(iou_type="bbox")

# Training loop
start_time = time.time()
log("Starting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    epoch_start = time.time()
    log(f"\nEpoch {epoch + 1}/{epochs}")

    for i, (images, targets) in enumerate(train_loader):
        step_start = time.time()
        images = images.to(device)
        preds = model(images)

        targets_encoded = [encode_targets(t) for t in targets]
        targets_batch = torch.stack(targets_encoded).to(device)

        loss = yolo_loss(preds, targets_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress = 100 * (i + 1) / len(train_loader)
        step_time = time.time() - step_start
        eta = step_time * (len(train_loader) - (i + 1))
        log(f"[{i + 1}/{len(train_loader)}] Loss: {loss.item():.4f} | Step: {step_time:.2f}s | ETA: {eta:.0f}s | {progress:.1f}%")

    avg_loss = total_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    log(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s - Avg loss: {avg_loss:.4f}")

    # Validation
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss = 0
        metric.reset()

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                preds = model(images)

                # 1) metrics
                preds_list   = postprocess_predictions(preds)
                targets_list = convert_targets_for_metrics(targets)
                metric.update(preds_list, targets_list)

                # 2) loss
                targets_enc  = [encode_targets(t) for t in targets]
                targets_batch= torch.stack(targets_enc).to(device)
                val_loss += yolo_loss(preds, targets_batch).item()

        results = metric.compute()
        val_avg = val_loss / len(val_loader)
        log(f"Val loss: {val_avg:.4f}")
        log(f"Val P: {results.get('precision',0):.4f}, "
            f"R: {results.get('recall',0):.4f}, "
            f"mAP@0.5: {results.get('map_50',0):.4f}, "
            f"mAP@0.5:0.95: {results.get('map',0):.4f}")

# Save model
torch.save(model.state_dict(), "yolo_custom_car_model.pth")
total_time = time.time() - start_time
log(f"\nModel saved as 'yolo_custom_car_model.pth'")
log(f"Training finished in {total_time/60:.2f} minutes")

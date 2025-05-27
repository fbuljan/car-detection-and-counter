import torch
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

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

for imgs, targets in loader:
    print(imgs.shape)
    for i, target in enumerate(targets):
        print(f"Slika {i} – broj oznaka: {target.shape[0]}")
    break
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class CarDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_size=(384, 640), transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size  # (H, W)
        self.transform = transform

        # List all image filenames
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(self.labels_dir, image_name.replace('.jpg', '.txt'))

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize image
        image = TF.resize(image, self.image_size)
        image_tensor = TF.to_tensor(image)  # [0, 1], shape: (3, H, W)

        # Load labels
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    # Format: class x_center y_center width height
                    targets.append(parts)

        targets = torch.tensor(targets, dtype=torch.float32)  # (N, 5)

        return image_tensor, targets

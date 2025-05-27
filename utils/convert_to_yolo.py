import os
import shutil
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from PIL import Image

# Paths
RAW_DIR = "dataset_raw"
TRAIN_IMG_DIR = os.path.join(RAW_DIR, "training_images")
TEST_IMG_DIR = os.path.join(RAW_DIR, "testing_images")
CSV_PATH = os.path.join(RAW_DIR, "car_labels.csv")  # Change if needed
OUTPUT_DIR = "dataset"

# Create YOLO folder structure
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# Read annotations CSV
df = pd.read_csv(CSV_PATH)

# Group boxes by image
annotations = {}
for _, row in df.iterrows():
    img = row['image']
    bbox = row[['xmin', 'ymin', 'xmax', 'ymax']].tolist()
    annotations.setdefault(img, []).append(bbox)

# Unique image names
all_images = list(annotations.keys())
random.shuffle(all_images)

# Split into train (70%) and temp (30%)
train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42)

# Split temp into val and test (15% each of total)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

def convert_bbox(size, box):
    w_img, h_img = size
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0 / w_img
    y_center = (y_min + y_max) / 2.0 / h_img
    width = (x_max - x_min) / w_img
    height = (y_max - y_min) / h_img
    return [x_center, y_center, width, height]

def process_images(image_list, split):
    for img_name in image_list:
        src_img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        if not os.path.exists(src_img_path):
            src_img_path = os.path.join(TEST_IMG_DIR, img_name)
            if not os.path.exists(src_img_path):
                print(f"⚠️ Image not found: {img_name}")
                continue

        with Image.open(src_img_path) as img:
            width, height = img.size

        dst_img_path = os.path.join(OUTPUT_DIR, "images", split, img_name)
        shutil.copy2(src_img_path, dst_img_path)

        label_path = os.path.join(OUTPUT_DIR, "labels", split, img_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            for bbox in annotations[img_name]:
                yolo_box = convert_bbox((width, height), bbox)
                f.write(f"0 {' '.join([f'{x:.6f}' for x in yolo_box])}\n")

# Apply to all splits
process_images(train_imgs, "train")
process_images(val_imgs, "val")
process_images(test_imgs, "test")

print("✅ Done: Dataset converted with 70% train, 15% val, 15% test.")

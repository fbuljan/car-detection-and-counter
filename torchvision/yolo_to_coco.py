import os
import json
import glob
from PIL import Image

def convert_yolo_to_coco(images_dir, labels_dir, output_json_path, category_name="car"):
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 3, "name": category_name}]
    }

    ann_id = 0
    for img_id, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))

        # Load image size
        with Image.open(img_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 5:
                    continue
                class_id, cx, cy, w, h = parts
                x = (cx - w / 2) * width
                y = (cy - h / 2) * height
                abs_w = w * width
                abs_h = h * height

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 3,
                    "bbox": [x, y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
                ann_id += 1

    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"✅ COCO file saved to {output_json_path} ({len(coco['annotations'])} annotations)")


if __name__ == "__main__":
    convert_yolo_to_coco(
        images_dir="datasets/cars/images/train",
        labels_dir="datasets/cars/labels/train",
        output_json_path="datasets/cars/coco_annotations_train.json"
    )

    convert_yolo_to_coco(
        images_dir="datasets/cars/images/val",
        labels_dir="datasets/cars/labels/val",
        output_json_path="datasets/cars/coco_annotations_val.json"
    )

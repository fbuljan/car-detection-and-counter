import os
import cv2
import argparse
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF

CAR_CLASS_ID = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fasterrcnn_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)  
    model.to(device)
    model.eval()
    return model

def detect_and_count_cars(model, image_path, output_path, score_threshold=0.5):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[WARNING] Ne mogu učitati sliku: {image_path}")
        return 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = TF.to_tensor(img_rgb).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    car_count = 0
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score >= score_threshold and label.item() == CAR_CLASS_ID:
            car_count += 1
            x1, y1, x2, y2 = map(int, box.tolist())

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"car {score:.2f}"
            cv2.putText(img_bgr, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(img_bgr, f"Cars detected: {car_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_bgr)
    print(f"[INFO] Procesirana: {image_path} --> {output_path} (Cars: {car_count})")

    return car_count

def process_folder(model, input_dir, output_dir):
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_ext:
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)
            detect_and_count_cars(model, in_path, out_path)

def main():
    parser = argparse.ArgumentParser(description="Detect and count cars using Faster R-CNN.")
    parser.add_argument('--output', type=str, default='count_cars_outputs/count_cars_outputs_fast_r_cnn_original',
                        help='Output folder')
    args = parser.parse_args()

    print(f"[INFO] Učitavanje Faster R-CNN COCO pretrained modela ...")
    model = load_fasterrcnn_model()

    base_input_dir = "datasets/cars/images"
    base_output_dir = args.output

    for subset in ["train", "val", "test"]:
        input_dir = os.path.join(base_input_dir, subset)
        output_dir = os.path.join(base_output_dir, subset)
        print(f"[INFO] Procesiranje foldera {input_dir}")
        if os.path.exists(input_dir):
            process_folder(model, input_dir, output_dir)
        else:
            print(f"[WARNING] Folder ne postoji: {input_dir}")

if __name__ == "__main__":
    main()
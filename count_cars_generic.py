import os
import cv2
import argparse
import torch
from torchvision.transforms import functional as TF

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from torchvision.models.detection import fasterrcnn_resnet50_fpn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_with_yolo(model_path, image_path, output_path):
    if YOLO is None:
        raise ImportError("Ultralytics YOLO not found. Please install it with `pip install ultralytics`")

    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Ne mogu učitati sliku: {image_path}")
        return 0

    results = model(img)[0]
    car_count = 0

    for box in results.boxes:
        class_id = int(box.cls.item())
        conf = box.conf.item()
        if class_id == 2:  
            car_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"car {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.putText(img, f"Cars detected: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)

    output_dir = os.path.dirname(output_path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"[INFO] Procesirana: {image_path} --> {output_path} (Cars: {car_count})")
    return car_count

def detect_with_fasterrcnn(model, image_path, output_path, car_class_id):
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
        if score >= 0.5 and label.item() == car_class_id:
            car_count += 1
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"car {score:.2f}"
            cv2.putText(img_bgr, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(img_bgr, f"Cars detected: {car_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    output_dir = os.path.dirname(output_path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, img_bgr)
    print(f"[INFO] Procesirana: {image_path} --> {output_path} (Cars: {car_count})")
    return car_count

def load_fasterrcnn_original():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model

def load_fasterrcnn_finetuned(weights_path, num_classes=2):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Generic car detection using multiple models.")
    parser.add_argument('--model-type', type=str, choices=['yolo', 'fasterrcnn-original', 'fasterrcnn-finetuned'], required=True, help='Model type')
    parser.add_argument('--model-path', type=str, help='Path to model weights (YOLO or fine-tuned Faster R-CNN)')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output-path', type=str, default='output/annotated.jpg', help='Path to save output image')
    args = parser.parse_args()

    if args.model_type == 'yolo':
        model_path = args.model_path or "yolov8l.pt"
        detect_with_yolo(model_path, args.image_path, args.output_path)

    elif args.model_type == 'fasterrcnn-original':
        model = load_fasterrcnn_original()
        detect_with_fasterrcnn(model, args.image_path, args.output_path, car_class_id=3)

    elif args.model_type == 'fasterrcnn-finetuned':
        if not args.model_path:
            print("[ERROR] You must provide --model-path for fine-tuned Faster R-CNN")
            return
        model = load_fasterrcnn_finetuned(args.model_path, num_classes=2)
        detect_with_fasterrcnn(model, args.image_path, args.output_path, car_class_id=1)

if __name__ == "__main__":
    main()

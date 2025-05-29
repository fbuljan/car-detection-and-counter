import os
import cv2
import argparse
from ultralytics import YOLO


COCO_CLASSES = {2: "car"}

def detect_and_count_cars(model, image_path, output_path):
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


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, img)
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
    parser = argparse.ArgumentParser(description="Detect and count cars on images.")
    parser.add_argument('--model', type=str, default='yolov8l.pt', help='Path to YOLO model')
    parser.add_argument('--output', type=str, default='count_cars_outputs/count_cars_outputs_yolo', help='Output folder')

    args = parser.parse_args()


    print(f"[INFO] Učitavanje modela {args.model} ...")
    model = YOLO(args.model)

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
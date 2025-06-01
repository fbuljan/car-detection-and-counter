import os
import cv2
import argparse
import torch
from torchvision.transforms import functional as TF
from collections import defaultdict

from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def run_video_detection(args):
    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = DeepSort(max_age=30)

    if args.model_type == 'yolo':
        model = YOLO(args.model_path or "yolov8l.pt")
    elif args.model_type == 'fasterrcnn-original':
        model = load_fasterrcnn_original()
    elif args.model_type == 'fasterrcnn-finetuned':
        model = load_fasterrcnn_finetuned(args.model_path, num_classes=2)
    else:
        raise ValueError("Unknown model type")

    unique_ids = set()
    frame_num = 0
    last_tracks = []  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        detections = []

        if frame_num == 1 or frame_num % 4 == 0:
            print(f"[INFO] Processing frame {frame_num}...")


            if args.model_type == 'yolo':
                results = model(frame)[0]
                for box in results.boxes:
                    if int(box.cls.item()) == 2:  
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf.item())
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'car'))
            else:
                img_tensor = TF.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
                with torch.no_grad():
                    outputs = model([img_tensor])[0]

                for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
                    if score >= 0.5 and (
                        (args.model_type == 'fasterrcnn-original' and label.item() == 3) or
                        (args.model_type == 'fasterrcnn-finetuned' and label.item() == 1)
                    ):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        detections.append(([x1, y1, x2 - x1, y2 - y1], float(score.item()), 'car'))

            last_tracks = tracker.update_tracks(detections, frame=frame)


        for track in last_tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            unique_ids.add(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        cv2.putText(frame, f"Frame: {frame_num}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Output video saved to: {args.output_path}")
    print(f"[INFO] Total unique vehicles detected: {len(unique_ids)}")


def main():
    parser = argparse.ArgumentParser(description="Vehicle tracking and counting with Deep SORT.")
    parser.add_argument('--model-type', choices=['yolo', 'fasterrcnn-original', 'fasterrcnn-finetuned'], required=True)
    parser.add_argument('--model-path', type=str, help='Model weights (for YOLO or finetuned Faster R-CNN)')
    parser.add_argument('--video-path', type=str, required=True, help='Input video path')
    parser.add_argument('--output-path', type=str, default='output/tracked.mp4', help='Output video path')
    args = parser.parse_args()

    run_video_detection(args)


if __name__ == "__main__":
    main()

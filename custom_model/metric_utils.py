import torch

def postprocess_predictions(preds, conf_threshold=0.5):
    B, H, W, _ = preds.shape
    pred_list = []

    for b in range(B):
        boxes, scores, labels = [], [], []

        for i in range(H):
            for j in range(W):
                pred = preds[b, i, j]
                obj_conf = torch.sigmoid(pred[4])
                if obj_conf < conf_threshold:
                    continue

                cx = (j + torch.sigmoid(pred[0])) * (640 / W)
                cy = (i + torch.sigmoid(pred[1])) * (384 / H)
                w = torch.exp(pred[2]) * (640 / W)
                h = torch.exp(pred[3]) * (384 / H)
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

                boxes.append(torch.tensor([x1, y1, x2, y2]))
                scores.append(obj_conf)
                labels.append(torch.tensor(0))  # samo 'car'

        if boxes:
            pred_list.append({
                "boxes": torch.stack(boxes),
                "scores": torch.stack(scores),
                "labels": torch.stack(labels)
            })
        else:
            pred_list.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64)
            })

    return pred_list

def convert_targets_for_metrics(targets):
    target_list = []

    for t in targets:
        if t.numel() == 0:
            target_list.append({
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64)
            })
            continue

        cx, cy = t[:, 1] * 640, t[:, 2] * 384
        w, h = t[:, 3] * 640, t[:, 4] * 384
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        target_list.append({
            "boxes": torch.stack([x1, y1, x2, y2], dim=1),
            "labels": t[:, 0].long()
        })

    return target_list

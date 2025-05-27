import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: raw scores, targets: 0/1
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        w = self.alpha * (1 - pt) ** self.gamma
        loss = w * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def yolo_loss(preds, targets,
                       box_weight=5.0, iou_weight=2.0,
                       obj_weight=1.0, cls_weight=1.0,
                       focal_alpha=0.25, focal_gamma=2.0):
    """
    preds: (B, S, S, 6) – x, y, w, h, obj_logit, cls_logit
    targets: (B, S, S, 6) – x, y, w, h, obj_target, cls_target
    """
    # separate predictions and targets
    pred_xywh = preds[..., :4]
    pred_obj_logits = preds[..., 4]
    pred_cls_logits = preds[..., 5]
    tgt_xywh = targets[..., :4]
    tgt_obj = targets[..., 4]
    tgt_cls = targets[..., 5]

    # masks
    obj_mask = tgt_obj == 1    # cells with object
    noobj_mask = tgt_obj == 0  # cells without object

    # 1) Coordinate MSE
    coord_loss = F.mse_loss(pred_xywh[obj_mask], tgt_xywh[obj_mask], reduction='mean') if obj_mask.any() else torch.tensor(0., device=preds.device)

    # 2) GIoU loss
    # convert (cx,cy,w,h) → (x1,y1,x2,y2)
    with torch.no_grad():
        pred_boxes = box_convert(pred_xywh[obj_mask], in_fmt='cxcywh', out_fmt='xyxy')
        tgt_boxes = box_convert(tgt_xywh[obj_mask], in_fmt='cxcywh', out_fmt='xyxy')
    if pred_boxes.numel() > 0:
        giou = generalized_box_iou(pred_boxes, tgt_boxes)
        giou_loss = (1 - giou).mean()
    else:
        giou_loss = torch.tensor(0., device=preds.device)

    # 3) Focal Loss for objectness
    focal_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    obj_loss = focal_fn(pred_obj_logits, tgt_obj)

    # 4) Focal Loss for class (only where there is an object)
    cls_loss = focal_fn(pred_cls_logits[obj_mask], tgt_cls[obj_mask]) if obj_mask.any() else torch.tensor(0., device=preds.device)

    # total loss
    total_loss = (box_weight * coord_loss +
                  iou_weight * giou_loss +
                  obj_weight * obj_loss +
                  cls_weight * cls_loss)
    return total_loss

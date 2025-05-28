import torch
import torch.nn.functional as F

def yolo_loss(preds, targets, coord_weight=5.0, noobj_weight=0.5):
    """
    preds: (B, S, S, 6) → [x, y, w, h, obj_logit, cls_logit]
    targets: (B, S, S, 6) → [x, y, w, h, obj, cls]
    """
    # ekstraktiramo sve
    pxywh = preds[..., :4]
    pobj = preds[..., 4]
    pcls = preds[..., 5]
    txywh = targets[..., :4]
    tobj = targets[..., 4]
    tcls = targets[..., 5]

    obj_mask = tobj == 1
    noobj_mask = tobj == 0

    # 1) coord loss (MSE only on positive cells)
    loss_xywh = F.mse_loss(pxywh[obj_mask], txywh[obj_mask]) if obj_mask.any() else 0.0

    # 2) objectness (BCE on all cells but downweighted on negatives)
    loss_obj = F.binary_cross_entropy_with_logits(pobj[obj_mask], tobj[obj_mask]) if obj_mask.any() else 0.0
    loss_noobj = F.binary_cross_entropy_with_logits(pobj[noobj_mask], tobj[noobj_mask])
    
    # 3) class (BCE only on positive cells)
    loss_cls = F.binary_cross_entropy_with_logits(pcls[obj_mask], tcls[obj_mask]) if obj_mask.any() else 0.0

    # ukupno
    total = coord_weight * loss_xywh \
            +      loss_obj \
            + noobj_weight * loss_noobj \
            +      loss_cls

    return total

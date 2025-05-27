import torch
import torch.nn.functional as F

def yolo_loss(preds, targets, lambda_coord=5.0, lambda_noobj=0.5):
    """
    preds:   (B, S, S, 6) → x, y, w, h, obj, class
    targets: (B, S, S, 6) → ground truth encoded grid
    """

    obj_mask = targets[..., 4] == 1    # where there is an object
    noobj_mask = targets[..., 4] == 0  # where there is no object

    # --- Loss for coordinates (x, y, w, h)
    coord_loss = F.mse_loss(preds[..., 0:2][obj_mask], targets[..., 0:2][obj_mask])  # x, y
    size_loss  = F.mse_loss(preds[..., 2:4][obj_mask], targets[..., 2:4][obj_mask])  # w, h

    # --- Loss for objectness
    obj_loss    = F.binary_cross_entropy_with_logits(preds[..., 4][obj_mask], targets[..., 4][obj_mask])
    noobj_loss  = F.binary_cross_entropy_with_logits(preds[..., 4][noobj_mask], targets[..., 4][noobj_mask])

    # --- Loss for class (1 class → BCE)
    class_loss = F.binary_cross_entropy_with_logits(preds[..., 5][obj_mask], targets[..., 5][obj_mask])

    total_loss = lambda_coord * (coord_loss + size_loss) + obj_loss + lambda_noobj * noobj_loss + class_loss
    return total_loss

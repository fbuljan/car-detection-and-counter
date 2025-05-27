import torch

def encode_targets(targets, grid_size=(24, 40), num_classes=1):
    S_h, S_w = grid_size
    encoded = torch.zeros((S_h, S_w, 5 + num_classes))  # (x, y, w, h, obj, class)

    for box in targets:
        cls, x, y, w, h = box

        grid_x = int(x * S_w)
        grid_y = int(y * S_h)

        local_x = x * S_w - grid_x
        local_y = y * S_h - grid_y

        if grid_y < S_h and grid_x < S_w:
            encoded[grid_y, grid_x, 0] = local_x
            encoded[grid_y, grid_x, 1] = local_y
            encoded[grid_y, grid_x, 2] = w
            encoded[grid_y, grid_x, 3] = h
            encoded[grid_y, grid_x, 4] = 1  # objectness
            encoded[grid_y, grid_x, 5] = 1  # class prob (1 because we have car only)

    return encoded

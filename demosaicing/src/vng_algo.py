import numpy as np
from tqdm import tqdm

directions = {
    'N': [-1, 0],
    'S': [1, 0],
    'E': [0, 1],
    'W': [0, -1],
    'NE': [-1, 1],
    'SE': [1, 1],
    'NW': [-1, -1],
    'SW': [1, -1]
}


def get_color_idx(rgb):
    idx = -1
    if rgb.any():
        idx = np.nonzero(rgb)[0][0]
    return idx


def get_grad(m, direction):
    up, right = direction
    grad = abs(sum(m[2 + up][2 + right] - m[2 - up][2 - right])) + \
           abs(sum(m[2 + 2 * up][2 + 2 * right] - m[2][2]))
    if up and right:
        grad += 0.5 * abs(sum(m[2 + 2 * up][2 + right] - m[2 + up][2])) + \
                0.5 * abs(sum(m[2 + up][2] - m[2][2 - right])) + \
                0.5 * abs(sum(m[2 + up][2 + 2 * right] - m[2][2 + right])) + \
                0.5 * abs(sum(m[2][2 + right] - m[2 - up][2]))
    else:
        directed_m = m[min(1, 2 + 2 * up): max(4, 2 + 2 * up + 1), min(1, 2 + 2 * right): max(4, 2 + 2 * right + 1)]
        if directed_m.shape != (4, 3, 3):
            directed_m = np.transpose(directed_m, (1, 0, 2))
        for i in range(2):
            grad += 0.5 * abs(sum(directed_m[i][0] - directed_m[i + 2][0])) + \
                    0.5 * abs(sum(directed_m[i][2] - directed_m[i + 2][2]))
    return grad


def get_rgb(m, direction, center_idx):
    rgb = np.zeros(3)
    up, right = direction

    center_vec = m[2][2] + m[2 + 2 * up][2 + 2 * right]
    rgb[center_idx] = 0.5 * sum(center_vec)

    next_idx = get_color_idx(m[2 + up][2 + right])
    if next_idx != -1:
        rgb[next_idx] = sum(m[2 + up][2 + right])

    if up and right:
        horizon_vec = m[2 + up][2] + m[2 + up][2 + 2 * right]
        horizon_idx = get_color_idx(horizon_vec)
        if horizon_idx != -1:
            rgb[horizon_idx] += 0.5 * sum(horizon_vec)

        vertical_vec = m[2][2 + right] + m[2 + 2 * up][2 + right]
        vertical_idx = get_color_idx(vertical_vec)
        if vertical_idx != -1:
            rgb[vertical_idx] += 0.5 * sum(vertical_vec)

        if center_idx != 1:
            rgb[1] *= 0.5
    else:
        if center_idx == 1:
            rb_others_vec = m[2 + right][2 + up] + m[2 + 2 * up + right][2 + up + 2 * right] + \
                            m[2 - right][2 - up] + m[2 + 2 * up - right][2 - up + 2 * right]
            rb_others_idx = get_color_idx(rb_others_vec)
            if rb_others_idx != -1:
                rgb[rb_others_idx] = 0.25 * sum(rb_others_vec)
        else:
            g_others_vec = m[2 + up + right][2 + up + right] + m[2 + up - right][2 - up + right]
            g_others_idx = get_color_idx(g_others_vec)
            if g_others_idx != -1:
                rgb[g_others_idx] = 0.5 * sum(g_others_vec)
    return rgb


def get_center(m, center_idx):
    matrix = np.copy(m)
    grads = {}
    for name in directions.keys():
        grads[name] = get_grad(matrix, directions[name])
    min_grad = min(grads.values())
    max_grad = max(grads.values())
    t = 1.5 * min_grad + 0.5 * (max_grad + min_grad)
    under_t_grads = {k: v for k, v in grads.items() if v <= t}

    rgb = np.zeros(3)
    for name in under_t_grads.keys():
        rgb += get_rgb(matrix, directions[name], center_idx)

    center_rgb = matrix[2][2]

    idx_others = list(x for x in range(3) if x != center_idx)
    for other_id in idx_others:
        if under_t_grads:
            center_rgb[other_id] = center_rgb[center_idx] + (rgb[other_id] - rgb[center_idx]) / len(under_t_grads)
    return center_rgb


def demosaicing_image(img_matrix):
    h, w, _ = img_matrix.shape
    new_img_matrix = np.zeros((h, w, 3), dtype=np.uint8)
    center_idx = get_color_idx(img_matrix[2][2])
    next_idx = get_color_idx(img_matrix[2][3])
    with tqdm(total=h * w, desc='VNG progress bar') as vng_progress_bar:
        for i in range(h - 5):
            cur_center = center_idx if not i % 2 else next_idx
            cur_next = next_idx if not i % 2 else next(x for x in range(3) if x != center_idx and x != next_idx)
            for j in range(w - 5):
                cur_idx = cur_center if not j % 2 else cur_next
                new_img_matrix[i + 2][j + 2] = get_center(img_matrix[i: i + 5, j: j + 5], cur_idx)
                vng_progress_bar.update(1)
    return new_img_matrix

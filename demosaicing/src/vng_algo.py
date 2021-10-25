import numpy as np

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
        if directed_m.shape != (4, 3):
            directed_m = directed_m.T
        for i in range(2):
            grad += 0.5 * abs(sum(directed_m[i][0] - directed_m[i + 2][0])) + \
                    0.5 * abs(sum(directed_m[i][2] - directed_m[i + 2][2]))
    return grad


def get_rgb(m, direction):
    rgb = np.zeros(3)
    up, right = direction

    idx_center = list(map(bool, m[2][2])).index(True)
    rgb[idx_center] = 0.5 * sum(m[2][2] + m[2 + 2 * up][2 + 2 * right])

    idx_next_to_center = list(map(bool, m[2 + up][2 + right])).index(True)
    rgb[idx_next_to_center] = m[2 + up][2 + right]

    last_idx = next(x for x in range(3) if x != idx_center and x != idx_next_to_center)
    directed_m = m[min(2, 2 + 2 * up): max(3, 2 + 2 * up + 1),
                   min(2, 2 + 2 * right): max(3, 2 + 2 * right + 1)].flatten()
    others = list(filter(lambda x: x[last_idx], directed_m))
    rgb[last_idx] = sum(others) / len(others)

    return rgb


def get_matrix_center(matrix):
    grads = {}
    for name in directions.keys():
        grads[name] = get_grad(matrix, directions[name])
    min_grad = min(grads.values())
    max_grad = max(grads.values())
    t = 1.5 * min_grad + 0.5 * (max_grad + min_grad)
    under_t_grads = {k: v for k, v in grads.items() if v < t}
    rgb = np.zeros(3)
    for name in under_t_grads.keys():
        rgb += get_rgb(matrix, directions[name])

    center_rgb = matrix[2][2]
    idx_center = list(map(bool, center_rgb)).index(True)
    idx_others = list(x for x in range(3) if x != idx_center)
    for other_id in idx_others:
        center_rgb[other_id] = center_rgb[idx_center] + (rgb[other_id] - rgb[idx_center]) / len(under_t_grads)
    return center_rgb

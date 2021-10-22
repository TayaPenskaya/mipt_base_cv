def get_grad(m, up, right):
    grad = abs(sum(m[2 + up][2 + right] - m[2 - up][2 - right])) + \
           abs(sum(m[2 + 2 * up][2 + 2 * right] - m[2][2]))
    if up and right:
        if m[2][2][1]:
            grad += abs(sum(m[2 + 2 * up][2 + right] - m[2][2 - right])) + \
                    abs(sum(m[2 + up][2 + 2 * right] - m[2 - up][2]))
        else:
            grad += abs(sum(m[2 + up][2] - m[2][2 - right])) / 2 + \
                    abs(sum(m[2 - up][2] - m[2][2 + right])) / 2 + \
                    abs(sum(m[2 + 2 * up][2 + right] - m[2 + up][2])) / 2 + \
                    abs(sum(m[2 + up][2 + 2 * right] - m[2][2 + right])) / 2
    else:
        grad += abs(sum(m[2 + up - right][2 + up + right] - m[2 - up - right][2 + up - right])) / 2 + \
                abs(sum(m[2 + up + right][2 - up + right] - m[2 - up + right][2 - up - right])) / 2 + \
                abs(sum(m[2 + 2 * up - right][2 + 2 * right + up] - m[2 - right][2 + up])) / 2 + \
                abs(sum(m[2 + 2 * up + right][2 + 2 * right - up] - m[2 + right][2 - up])) /2
    return grad



# def get_matrix_center(matrix):
#     grads = {}
#     grads['N'] = get_grad(matrix, 1, 0)

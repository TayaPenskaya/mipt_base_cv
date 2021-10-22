

def get_grad(m, up, right):
    print(
        m[2 + up][2 + right], m[2 - up][2 - right],
        m[2 + 2 * up][2 + 2 * right], m[2][2]
    )
    if up and right:
        print(
            m[2 + up][2], m[2][2 - right],
            m[2 - up][2], m[2][2 + right]
        )
        print(
            m[2 + 2 * up][2 + right], m[2 + up][2],
            m[2 + up][2 + 2 * right], m[2][2 + right],
        )
    else:
        print(
            m[2 + up - right][2 + up + right], m[2 - up - right][2 + up - right],
            m[2 + up + right][2 - up + right], m[2 - up + right][2 - up - right]
        )
        print(
            m[2 + 2 * up - right][2 + 2 * right + up], m[2 - right][2 + up],
            m[2 + 2 * up + right][2 + 2 * right - up], m[2 + right][2 - up]
        )

# def get_matrix_center(matrix):
#     grads = {}
#     grads['N'] = get_grad(matrix, 1, 0)


m = [['r1', 'g2', 'r3', 'g4', 'r5'],
     ['g6', 'b7', 'g8', 'b9', 'g10'],
     ['r11', 'g12', 'r13', 'g14', 'r15'],
     ['g16', 'b17', 'g18', 'b19', 'g20'],
     ['r21', 'g22', 'r23', 'g24', 'r25']]
get_grad(m, -1, -1)

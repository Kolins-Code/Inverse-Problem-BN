import numpy as np
import scipy.sparse as sp


def generate_single(dt, dx, end_t, end_x, func, save_path, val_path):
    size_x = (int) (end_x / dx)
    size_t = (int) (end_t / dt)

    grid_x = np.linspace(0, end_x, size_x + 1)
    grid_t = np.linspace(0, end_t, size_t + 1)

    start_func = func(grid_x)

    solution = start_func

    gamma = dt / (dx * dx)
    #matrix = sp.diags([-0.5 * gamma, 1 + gamma, -0.5 * gamma], [-1, 0, 1], shape=(size_x + 1, size_x + 1)).toarray()
    matrix = sp.diags([-1 * gamma, 1 + 2 * gamma, -1 * gamma], [-1, 0, 1], shape=(size_x + 1, size_x + 1)).toarray()
    matrix[0][0] = 1
    matrix[0][1] = 0
    matrix[-1][-1] = 1
    matrix[-1][-2] = 0

    #f_matrix = sp.diags([0.5 * gamma, 1 - gamma, 0.5 * gamma], [-1, 0, 1], shape=(size_x + 1, size_x + 1)).toarray()
    f_matrix = sp.diags([0, 1, 0], [-1, 0, 1], shape=(size_x + 1, size_x + 1)).toarray()
    f_matrix[0][0] = 0
    f_matrix[0][1] = 0
    f_matrix[-1][-1] = 0
    f_matrix[-1][-2] = 0

    for i in range(1, size_t + 1):
        f = np.dot(solution, f_matrix)
        solution = sp.linalg.spsolve(matrix, f)

    np.savetxt(val_path, start_func, delimiter=",", fmt="%.18f")
    np.savetxt(save_path, solution, delimiter=",", fmt="%.18f")

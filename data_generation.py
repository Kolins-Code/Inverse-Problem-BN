import numpy as np
import scipy.sparse as sp
from start_func_generation import *


def generate_single(dt, dx, end_t, end_x, func, save_path, val_path):
    size_x = (int) (end_x / dx)
    size_t = (int) (end_t / dt)

    grid_x = np.linspace(0, end_x, size_x + 1)
    grid_t = np.linspace(0, end_t, size_t + 1)

    start_func = func(grid_x)

    gamma = dt / (dx * dx)
    '''solution = start_func
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
        solution = sp.linalg.spsolve(matrix, f)'''
    
    coef_matrix = sp.diags([gamma, -1 - 2 * gamma, gamma], [-1, 0, 1], shape=(size_x - 1, size_x - 1))
    template = np.full((size_t + 1, size_t + 1), None)
    for i in range(size_t):
        template[i, i] = sp.identity(size_x - 1)
        template[i, i + 1] = coef_matrix
    template[size_t, 0] = sp.identity(size_x - 1)
    print(template)

    matrix = sp.bmat(template)
    print(matrix)
    f = np.zeros((size_t + 1) * (size_x - 1))
    print(f)
    f[size_t * (size_x - 1):] = start_func[1:size_x]
    solution = sp.linalg.spsolve(matrix, f)

    solution = np.insert(solution[size_t * (size_x - 1):], 0, 0)
    solution = np.append(solution, 0)

    np.savetxt(val_path, start_func, delimiter=",", fmt="%.18f")
    np.savetxt(save_path, solution, delimiter=",", fmt="%.18f")


def generate_harmonic(dt, dx, end_t, end_x, func, save_path, val_path, n):
    size_x = (int) (end_x / dx)
    grid_x = np.linspace(0, end_x, size_x + 1)

    def solution_func(x, t):
        return np.sin((np.pi * n * x) / end_x) * np.exp(-(np.pi * n / end_x) * (np.pi * n / end_x) * t)
    
    start_func = solution_func(grid_x, 0)

    np.savetxt(val_path, start_func, delimiter=",", fmt="%.18f")
    np.savetxt(save_path, solution_func(grid_x, end_t), delimiter=",", fmt="%.18f")

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from start_func_generation import *


def generate_single(dt, dx, end_t, end_x, func, save_path, val_path):
    size_x = (int) (end_x / dx)
    size_t = (int) (end_t / dt)

    grid_x = np.linspace(0, end_x, size_x + 1)
    grid_t = np.linspace(0, end_t, size_t + 1)

    start_func = func(grid_x)

    gamma = dt / (dx * dx)
    solution = start_func
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
    
    '''coef_matrix = sp.diags([gamma, -1 - 2 * gamma, gamma], [-1, 0, 1], shape=(size_x - 1, size_x - 1))
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
    solution = np.append(solution, 0)'''

    solution += np.random.normal(0, 1.e-2)

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

def generate_laplace(dx, end_x, skip, func1, func2, save_path, start_path, val_path):
    size_x = (int) (end_x / dx)
    grid_x = np.linspace(0, end_x, size_x + 1)

    func1 = func1(grid_x)
    func2 = func2(grid_x)

    coef_matrix = sp.diags([1, -4, 1], [-1, 0, 1], shape=(size_x - 1, size_x - 1))
    template = np.full((size_x + 1, size_x + 1), None)
    for i in range(1, size_x):
        template[i, i + 1] = sp.identity(size_x - 1)
        template[i, i] = coef_matrix
        template[i, i - 1] = sp.identity(size_x - 1)
    template[0, 0] = sp.identity(size_x - 1)
    template[size_x, size_x] = sp.identity(size_x - 1)

    matrix = sp.bmat(template)
    f = np.zeros((size_x + 1) * (size_x - 1))
    f[0:size_x - 1] = func1[1:size_x]
    f[size_x * (size_x - 1):] = func2[1:size_x]
    solution = sp.linalg.spsolve(matrix, f)

    
    solution = np.reshape(solution, (size_x + 1, size_x - 1)).transpose()
    solution = np.insert(solution, 0, [0], axis=0)
    solution = np.insert(solution, size_x, [0], axis=0)


    derivative_func = np.zeros(size_x + 1)
    for i in range(1, size_x):
        derivative_func[i] = (2 * solution[i, 1] + solution[i - 1, 0] + solution[i + 1, 0] - 4 * solution[i, 0]) / (2 * dx)

    derivative_func += np.random.normal(0, 0.001)
    
    np.savetxt(start_path, func1[::skip], delimiter=",", fmt="%.18f")
    np.savetxt(val_path, func2[::skip], delimiter=",", fmt="%.18f")
    np.savetxt(save_path, derivative_func[::skip], delimiter=",", fmt="%.18f")


#generate_laplace(0.01, np.pi, lambda x: np.zeros(x.shape), lambda x: np.sinh(np.pi) * np.sin(x), None, None)

def generate_laplace_coef(dx, end_x, skip, func, edge1_path, edge2_path, derivative_edge1_path, derivative_edge2_path, val_path):
    size_x = (int) (end_x / dx)
    grid_x = np.linspace(0, end_x, size_x + 1)

    func = func(grid_x)
    
    coef_matrix = sp.diags([1, -4, 1], [-1, 0, 1], shape=(size_x - 1, size_x - 1))
    template = np.full((size_x + 1, size_x + 1), None)
    for i in range(1, size_x):
        template[i, i + 1] = sp.identity(size_x - 1)
        template[i, i] = coef_matrix
        template[i, i - 1] = sp.identity(size_x - 1)
    template[0, 0] = sp.identity(size_x - 1)
    template[size_x, size_x] = sp.identity(size_x - 1)

    matrix = sp.bmat(template)
    f = np.zeros((size_x + 1) * (size_x - 1))
    for i in range(1, size_x):
        f[(size_x - 1) * i: (size_x - 1) * i + size_x - 1] = func[1:size_x] * (-np.abs(i*dx - end_x / 2) + end_x / 2) * dx * dx
    #f[0:size_x - 1] = -func[1:size_x]
    #f[size_x * (size_x - 1):] = -func[1:size_x]
    
    solution = sp.linalg.spsolve(matrix, f)

    
    solution = np.reshape(solution, (size_x + 1, size_x - 1)).transpose()
    solution = np.insert(solution, 0, [0], axis=0)
    solution = np.insert(solution, size_x, [0], axis=0)


    derivative_func1 = np.zeros(size_x + 1)
    derivative_func2 = np.zeros(size_x + 1)
    for i in range(1, size_x):
        derivative_func1[i] = (2 * solution[i, 1] + solution[i - 1, 0] + solution[i + 1, 0] - 4 * solution[i, 0]) / (2 * dx) - 0.5 * func[i] * 0
        derivative_func2[i] = (2 * solution[i, size_x - 1] + solution[i - 1, size_x] + solution[i + 1, size_x] - 4 * solution[i, size_x]) / (2 * dx) - 0.5 * func[i] * 0

    #derivative_func1[20:size_x - 19] += np.random.normal(0, 0.001)
    #derivative_func2[20:size_x - 19] += np.random.normal(0, 0.001)

    edge_func1 = solution[0:(size_x - 1), 0]
    edge_func2 = solution[0:(size_x - 1), size_x]

    np.savetxt(edge1_path, edge_func1[::skip], delimiter=",", fmt="%.18f")
    np.savetxt(edge2_path, edge_func2[::skip], delimiter=",", fmt="%.18f")
    np.savetxt(derivative_edge1_path, derivative_func1[::skip], delimiter=",", fmt="%.18f")
    np.savetxt(derivative_edge2_path, derivative_func2[::skip], delimiter=",", fmt="%.18f")
    np.savetxt(val_path, func[::skip], delimiter=",", fmt="%.18f")

#generate_laplace_coef(0.01, 1, 1, lambda x: np.ones(x.shape), None, None, None, None, None)
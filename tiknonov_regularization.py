import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize, newton


class RegularizationSolver:
    def __init__(self, dt, dx, end_t, end_x, error, input_path, save_path):
        self.dt = dt
        self.dx = dx
        self.end_t = end_t
        self.end_x = end_x
        
        self.size_x = (int) (self.end_x / self.dx)
        self.size_t = (int) (self.end_t / self.dt)
        self.grid_x = np.linspace(0, self.end_x, self.size_x + 1)

        self.error = error

        self.input_func = np.genfromtxt(input_path, delimiter=',')
        self.save_path = save_path
     
    def solve_problem(self, start_func):
        gamma = self.dt / (self.dx * self.dx)

        solution = start_func
        #matrix = sp.diags([-0.5 * gamma, 1 + gamma, -0.5 * gamma], [-1, 0, 1], shape=(size_x + 1, size_x + 1)).toarray()
        matrix = sp.diags([-1 * gamma, 1 + 2 * gamma, -1 * gamma], [-1, 0, 1], shape=(self.size_x + 1, self.size_x + 1)).toarray()
        matrix[0][0] = 1
        matrix[0][1] = 0
        matrix[-1][-1] = 1
        matrix[-1][-2] = 0

        #f_matrix = sp.diags([0.5 * gamma, 1 - gamma, 0.5 * gamma], [-1, 0, 1], shape=(size_x + 1, size_x + 1)).toarray()
        f_matrix = sp.diags([0, 1, 0], [-1, 0, 1], shape=(self.size_x + 1, self.size_x + 1)).toarray()
        f_matrix[0][0] = 0
        f_matrix[0][1] = 0
        f_matrix[-1][-1] = 0
        f_matrix[-1][-2] = 0

        for i in range(1, self.size_t + 1):
            f = np.dot(solution, f_matrix)
            solution = sp.linalg.spsolve(matrix, f)
        
        return solution

    def reg_func(self, grid_func, alpha):
        residual = np.sum((self.solve_problem(grid_func) - self.input_func)**2 * self.dx) #np.max(self.solve_problem(grid_func) - self.input_func)**2
        stabilizer = np.sum(grid_func**2 * self.dx) #np.max(grid_func)**2
        
        return residual + alpha * stabilizer

    def redisual_func(self, alpha):
        return np.sum((self.solve_problem(self.minimize(alpha)) - self.input_func)**2 * self.dx) - self.error**2
    
    def minimize(self, alpha):
        start_func = np.array([0] * (self.size_x + 1))
        minimization = minimize(self.reg_func, start_func, args=(alpha), options={'disp': True}, method="BFGS", tol=1.e-100)

        return minimization.x

    def solve(self):
        #alpha = newton(self.redisual_func, 1.e-7)
        #print(alpha)

        np.savetxt(self.save_path, self.minimize(1.e-2), delimiter=",", fmt="%.18f")

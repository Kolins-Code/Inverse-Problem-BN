import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

dt = 0.001
dx = 0.1
end_x = np.pi
end_t = 0.01
size_x = (int) (end_x / dx)
size_t = (int) (end_t / dt)

grid_x = np.linspace(0, end_x, size_x + 1)
grid_t = np.linspace(0, end_t, size_t + 1)

start_func = np.sin(2 * grid_x)
print(start_func)

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

    '''f = np.dot(solution[1:-1], f_matrix) 
    f[0] = f[0]+0.5*gamma*(0 + 0)
    f[-1] = B[-1]+0.5*gamma*(0 + 0)
    solution[1:-1] = np.linalg.solve(matrix, f)'''

#print(solution)
plt.plot(grid_x, solution)
plt.plot(grid_x, np.sin(2 * grid_x) * np.exp(-4 * end_t), c='r')
plt.show()

NAME = "data.csv"
np.savetxt(NAME, solution, delimiter=",", fmt="%.18f")

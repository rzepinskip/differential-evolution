import numpy as np

from diff_evolution.cec17_functions import cec17_test_func

# x: Solution vector
x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# nx: Number of dimensions
nx = 10

# mx: Number of objective functions
mx = 1

# func_num: Function number
func_num = 1


f = cec17_test_func(x, nx, mx, func_num)
print(f)

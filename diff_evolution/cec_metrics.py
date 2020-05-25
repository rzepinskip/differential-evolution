from diff_evolution.algo import ConstantDE, DifferentialEvolution
from diff_evolution.cec17_functions import cec17_test_func
from diff_evolution.algo_control import AlgorithmControl


import time

MAX_FES_FACTOR = 10000
TARGET_VALUE_FACTOR = 100
BOUNDS_1D = [(-100, 100)]


def run_problem(de: DifferentialEvolution, dims: int, func_num: int):
    max_fes = MAX_FES_FACTOR * dims
    bounds = BOUNDS_1D * dims
    target_value = TARGET_VALUE_FACTOR * func_num

    def call_cec(x):
        fitness = cec17_test_func(x, dims=dims, func_num=func_num)
        return fitness[0]

    algo_control = AlgorithmControl(call_cec, max_fes, dims, target_value)

    de.run(algo_control, bounds)

    return algo_control.error()


for func_num in [1, 29]:
    print(f'Tests for function {func_num}')
    for dims in [10, 30, 50]:
        print(f'Test started... Dims = {dims}')
        algo = ConstantDE(seed=44)
        start = time.time()
        err = run_problem(algo, dims, func_num)
        end = time.time()
        print(f'Error: {err}')
        print(f'Execution time: {end - start}s')
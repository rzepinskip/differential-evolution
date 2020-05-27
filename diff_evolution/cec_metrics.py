from diff_evolution.algo import ConstantDE, DifferentialEvolution
from diff_evolution.cec17_functions import cec17_test_func
from diff_evolution.algo_control import AlgorithmControl, RECORDING_POINTS


import time
import itertools
from multiprocessing import Pool
import numpy as np
from statistics import median, mean, stdev
import os
import json

MAX_FES_FACTOR = 10000
TARGET_VALUE_FACTOR = 100
BOUNDS_1D = [(-100, 100)]
PROCESSES_NUM = 4


def run_single_problem(de: DifferentialEvolution, dims: int, func_num: int, max_fes = None):
    if max_fes is None:
        max_fes = MAX_FES_FACTOR * dims
    bounds = BOUNDS_1D * dims
    target_value = TARGET_VALUE_FACTOR * func_num

    def call_cec(x):
        fitness = cec17_test_func(x, dims=dims, func_num=func_num)
        return fitness[0]

    algo_control = AlgorithmControl(call_cec, max_fes, dims, target_value)

    de.run(algo_control, bounds)

    algo_control.fill_up_recorder_values()

    return algo_control.recorded_values, algo_control.error()

def run_process(x):
    dims, func_num = x
    algo = ConstantDE()
    res = run_single_problem(algo, dims, func_num)
    return res

def run_multi_problems(problems):
    with Pool(PROCESSES_NUM) as p:
        resulsts = p.map(run_process, problems)
        return resulsts

def generate_output(algo_results, dims, func_num, output_path):
    res_table = np.zeros((len(RECORDING_POINTS), len(algo_results)))
    for i, algo_result in enumerate(algo_results):
        res_table[:,  i] = algo_result[0]

    errors = [el[1] for el in algo_results]
    metrics = {
        'best': min(errors),
        'worst': max(errors),
        'median': median(errors),
        'mean': mean(errors),
        'std': stdev(errors)
    }

    np.savetxt(os.path.join(output_path, f'{func_num}_{dims}.txt'), res_table, delimiter=',')

    with open(os.path.join(output_path, f'metrics_{func_num}_{dims}.json'), 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    for dims in [10, 30, 50]:
        for func_num in range(1, 30):
            print(f'Running test for function {func_num}, {dims} dims.')
            res = run_multi_problems([(dims, func_num)] * 51)
            generate_output(res, 10, 1, '.')
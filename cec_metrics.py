import itertools
import json
import math
import os
import time
from functools import partial
from multiprocessing import Pool
from statistics import mean, median, stdev

import numpy as np

from diff_evolution.algo import ConstantDE, DifferentialEvolution
from diff_evolution.algo_control import RECORDING_POINTS, AlgorithmControl
from diff_evolution.cec17_functions import cec17_test_func

MAX_FES_FACTOR = 10000
TARGET_VALUE_FACTOR = 100
BOUNDS_1D = [(-100, 100)]
PROCESSES_NUM = 4
COMPLEXITY_MEASURE_FUNCTION_NUM = 18


def run_single_problem(problem, de: DifferentialEvolution, max_fes=None):
    dims, func_num = problem
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


def run_multi_problems(algorithm, problems):
    run_process = partial(run_single_problem, de=algorithm)
    with Pool(PROCESSES_NUM) as p:
        resulsts = p.map(run_process, problems)
        return resulsts


def generate_output(algo_results, dims, func_num, output_path):
    res_table = np.zeros((len(RECORDING_POINTS), len(algo_results)))
    for i, algo_result in enumerate(algo_results):
        res_table[:, i] = algo_result[0]

    errors = [el[1] for el in algo_results]
    metrics = {
        "best": min(errors),
        "worst": max(errors),
        "median": median(errors),
        "mean": mean(errors),
        "std": stdev(errors),
    }

    np.savetxt(
        os.path.join(output_path, f"{func_num}_{dims}.txt"), res_table, delimiter=","
    )

    with open(os.path.join(output_path, f"metrics_{func_num}_{dims}.json"), "w") as f:
        json.dump(metrics, f)


def t0_function():
    for _ in range(1000000):
        x = 0.55  # changed!!! TODO: write about it in the report
        x = x + x
        x = x / 2
        x = x * x
        x = math.sqrt(x)
        x = math.log(x)
        x = math.exp(x)
        x = x / (x + 2)


def t1_function(dims):
    v = np.random.rand(dims)

    min_b, max_b = BOUNDS_1D[0][0], BOUNDS_1D[0][1]
    diff = np.fabs(min_b - max_b)
    v_denorm = min_b + v * diff

    for _ in range(200000):
        cec17_test_func(v_denorm, dims, func_num=COMPLEXITY_MEASURE_FUNCTION_NUM)


def t2_function(algorithm, dims):
    run_single_problem((dims, COMPLEXITY_MEASURE_FUNCTION_NUM), algorithm, 200000)


def measure_performance(algorithm, output_path):
    for dims in [10, 30, 50]:
        for func_num in range(1, 31):
            if func_num == 2:
                continue
            print(f"Running test for function {func_num}, {dims} dims.")
            res = run_multi_problems(algorithm, [(dims, func_num)] * 51)
            generate_output(res, dims, func_num, output_path)


def measure_complexity(algorithm, output_path):
    dims = [10, 30, 50]
    res_table = np.zeros((len(dims), 4))
    for i, dims in enumerate(dims):
        start = time.time()
        t0_function()
        end = time.time()
        t0 = end - start

        start = time.time()
        t1_function(dims)
        end = time.time()
        t1 = end - start

        t2s = []
        for _ in range(5):
            start = time.time()
            t2_function(algorithm, dims)
            end = time.time()
            t2s.append(end - start)

        t2prim = mean(t2s)

        res_table[i, :] = [t0, t1, t2prim, (t2prim - t1) / t0]

        np.savetxt(
            os.path.join(output_path, "complexity.txt"), res_table, delimiter=","
        )


if __name__ == "__main__":
    de = ConstantDE()

    measure_performance(de, ".")
    measure_complexity(de, ".")

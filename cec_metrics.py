import csv
import itertools
import json
import math
import os
import time
from functools import partial
from multiprocessing import Pool
from statistics import mean, median, stdev

import click
import numpy as np

from diff_evolution.algo import (
    ConstantDE,
    ConstantSuccessRuleDE,
    DifferentialEvolution,
    RandomSuccessRuleDE,
    init_population_uniform,
)
from diff_evolution.algo_control import RECORDING_POINTS, AlgorithmControl
from diff_evolution.cec17_functions import cec17_test_func

MAX_FES_FACTOR = 10000
TARGET_VALUE_FACTOR = 100
BOUNDS_1D = [(-100, 100)]
PROCESSES_NUM = 4


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

    de.run(algo_control, bounds, init_population_uniform)

    algo_control.fill_up_recorder_values()

    return algo_control.recorded_values, algo_control.error()


def run_multi_problems(algorithm, problems):
    run_process = partial(run_single_problem, de=algorithm)
    with Pool(PROCESSES_NUM) as p:
        resulsts = p.map(run_process, problems)
        return resulsts


def generate_output(algorithm, algo_results, dims, func_num, output_path):
    res_table = np.zeros((len(RECORDING_POINTS), len(algo_results)))
    for i, algo_result in enumerate(algo_results):
        res_table[:, i] = algo_result[0]

    errors = [el[1] for el in algo_results]
    metrics = {
        "func.": func_num,
        "best": f"{min(errors):.2E}",
        "worst": f"{max(errors):.2E}",
        "median": f"{median(errors):.2E}",
        "mean": f"{mean(errors):.2E}",
        "std": f"{stdev(errors):.2E}",
    }

    np.savetxt(
        os.path.join(
            output_path, f"{algorithm.__class__.__name__}_{func_num}_{dims}.txt"
        ),
        res_table,
        delimiter=",",
    )

    save_metrics_to_csv(
        os.path.join(output_path, f"{algorithm.__class__.__name__}_metrics_{dims}.csv"),
        metrics,
    )


def save_metrics_to_csv(file_path, metrics: dict):
    fieldnames = ["func.", "best", "worst", "median", "mean", "std"]

    write_header = False
    if not os.path.isfile(file_path):
        write_header = True

    with open(file_path, mode="a", newline="\n") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def measure_performance(algorithm, output_path, dimensions = [10, 30, 50], functions = range(1, 31)):
    for dims in dimensions:
        for func_num in range(1, 31):
            if func_num == 2:
                continue
            print(f"Running test for function {func_num}, {dims} dims.")
            res = run_multi_problems(algorithm, [(dims, func_num)] * 51)
            generate_output(algorithm, res, dims, func_num, output_path)


@click.command()
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Output directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--algo", "-a", required=True, help="Algorithm version name (class name)", type=str
)
@click.option(
    "--dims", "-d", help="Dimensions to be tested", type=int
)
def run_measurements(output_dir, algo, dims):
    alogrithms = {
        ConstantDE.__name__: ConstantDE,
        ConstantSuccessRuleDE.__name__: ConstantSuccessRuleDE,
        RandomSuccessRuleDE.__name__: RandomSuccessRuleDE,
    }

    de = alogrithms[algo]()

    if dims is not None:
        measure_performance(de, output_dir, dims)
    else:
        measure_performance(de, output_dir)


if __name__ == "__main__":
    run_measurements()

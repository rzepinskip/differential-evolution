from diff_evolution.algo import DifferentialEvolution
from diff_evolution.cec17_functions import cec17_test_func

MAX_FES_FACTOR = 10000
TARGET_VALUE_FACTOR = 100
BOUNDS_1D = [-100, 100]


def run_problem(de: DifferentialEvolution, dims: int, func_num: int):
    max_fes = MAX_FES_FACTOR * dims
    bounds = BOUNDS_1D * dims

    # de.run()
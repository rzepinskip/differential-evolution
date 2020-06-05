import math
from collections import deque

import numpy as np
import pytest

from diff_evolution.algo import (
    ConstantDE,
    ConstantSuccessRuleDE,
    RandomSuccessRuleDE,
    init_population_uniform,
)
from diff_evolution.algo_control import AlgorithmControl
from diff_evolution.cec17_functions import cec17_test_func

EPSILON = 1e-01
SEED = 44

tested_algos = [
    ConstantDE,
    ConstantSuccessRuleDE,
    RandomSuccessRuleDE,
]


@pytest.fixture(params=tested_algos)
def algo_version(request):
    return request.param


@pytest.mark.parametrize(
    "func, bounds, expected",
    [(lambda x: x ** 2, [(-5, 5)], 0), (lambda x: (x - 1) ** 2 + 3, [(-5, 5)], 1.0)],
)
def test_basic_functions(algo_version, func, bounds, expected):
    algo = algo_version(seed=SEED)

    algo_control = AlgorithmControl(func, 1000, 1, expected)

    # https://stackoverflow.com/questions/2138873/cleanest-way-to-get-last-item-from-python-iterator
    result = algo.run(algo_control, bounds, init_population_uniform)

    assert np.allclose(result, np.array(expected), atol=EPSILON)


def test_alpine(algo_version):
    algo = algo_version(seed=SEED)
    bounds = [(-5, 5)] * 2

    def alpine_one(X):
        return sum([abs(x * np.sin(x) + 0.1 * x) for x in X])

    algo_control = AlgorithmControl(alpine_one, 1000, 1, 0)

    result = algo.run(algo_control, bounds, init_population_uniform)

    assert np.allclose(result, np.array([0, 0]), atol=0.2)


def test_cec(algo_version):
    dims = 2
    bounds = [(-100, 100)] * dims

    def call_cec(x):
        fitness = cec17_test_func(x, dims=len(bounds), func_num=1)
        return fitness[0]

    algo = algo_version(seed=SEED)
    algo_control = AlgorithmControl(call_cec, 100000, 1, 100)

    result = algo.run(algo_control, bounds, init_population_uniform, population_size=20)

    # values taken from shift_data_1.txt
    assert np.allclose(
        result,
        np.array([-5.5276398498228005e01, -7.0429559718086182e01]),
        atol=EPSILON,
    )

import math
from collections import deque

import numpy as np
import pytest

from diff_evolution.algo import ConstantDE
from diff_evolution.cec17_functions import cec17_test_func

EPSILON = 1e-01
SEED = 44
ITERATIONS = 100


@pytest.mark.parametrize(
    "func, bounds, expected", [(lambda x: x ** 2, [(-5, 5)], 0)],
)
def test_basic_functions(func, bounds, expected):
    algo = ConstantDE(iterations=ITERATIONS, seed=SEED)

    # https://stackoverflow.com/questions/2138873/cleanest-way-to-get-last-item-from-python-iterator
    result = deque(algo.run(func, bounds), maxlen=1).pop()

    assert np.allclose(result, np.array(expected), atol=EPSILON)


def test_alpine():
    algo = ConstantDE(iterations=ITERATIONS, seed=SEED)
    bounds = [(-5, 5)] * 2

    def alpine_one(X):
        return sum([abs(x * np.sin(x) + 0.1 * x) for x in X])

    result = deque(algo.run(alpine_one, bounds), maxlen=1).pop()

    assert np.allclose(result, np.array([0, 0]), atol=0.2)


def test_cec():
    dims = 2
    algo = ConstantDE(iterations=ITERATIONS, seed=SEED)
    bounds = [(-100, 100)] * dims

    def call_cec(x):
        fitness = cec17_test_func(x, dims=len(bounds), func_num=1)
        return fitness[0]

    result = deque(algo.run(call_cec, bounds), maxlen=1).pop()

    assert np.allclose(result, np.array([0, 0]), atol=100)

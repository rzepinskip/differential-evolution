import math
from collections import deque

import numpy as np
import pytest

from diff_evolution.algo import ConstantDE

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

    def alpine_one(X):
        return sum([abs(x * np.sin(x) + 0.1 * x) for x in X])

    result = deque(algo.run(alpine_one, [(-5, 5)] * 2), maxlen=1).pop()

    assert np.allclose(result, np.array([0, 0]), atol=0.2)

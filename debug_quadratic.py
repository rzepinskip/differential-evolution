import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from diff_evolution.algo import (
    ConstantDE,
    ConstantSuccessRuleDE,
    ConstantSuccessRuleDEWithClip,
    RandomSuccessRuleDE,
    RandomSuccessRuleDEWithClip,
    init_population_uniform,
)
from diff_evolution.algo_control import AlgorithmControl
from diff_evolution.cec17_functions import cec17_test_func

MAX_FES_FACTOR = 1000
TARGET_VALUE_FACTOR = 100
BOUNDS_1D = [(-100, 100)]

dims = 10
max_fes = MAX_FES_FACTOR * dims
bounds = BOUNDS_1D * dims

linestyles = ["-", "--", "-.", ":"]
fig, ax = plt.subplots(2)


def init_population(population_size, bounds):
    bounds = [(9, 11)] * dims
    dimensions = len(bounds)
    pop_norm = np.random.rand(population_size, dimensions)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop_norm * diff

    return pop


for algo_version, linestyle in zip(
    [
        ConstantDE,
        # ConstantSuccessRuleDE,
        ConstantSuccessRuleDEWithClip,
        # RandomSuccessRuleDEWithClip,
        # RandomSuccessRuleDE,
    ],
    linestyles,
):
    for retry in range(5):
        algo = algo_version()

        def call_cec(x):
            return np.sum(x ** 2)

        target_value = 0
        algo_control = AlgorithmControl(call_cec, max_fes, dims, target_value)
        best_point = algo.run(algo_control, bounds, init_population)

        x = list(range(len(algo.population_history)))
        y = []
        for population in algo.population_history:
            y += [min([call_cec(member) for member in population])]

        error = f"{abs(call_cec(best_point) - target_value):.2E}"
        ax[0].plot(
            x,
            y,
            label=f"{algo_version.__name__}-{retry} ({error})",
            linestyle=linestyle,
        )
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Error")

        if algo_version != ConstantDE:
            ax[1].plot(
                list(range(len(algo.population_history))),
                algo.mutation_factor_history,
                label=f"{algo_version.__name__}-{retry} ({error})",
                linestyle=linestyle,
            )
            ax[1].set_ylim(0, 10)
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Mutation factor")
ax[0].legend()
ax[1].legend()
plt.show()

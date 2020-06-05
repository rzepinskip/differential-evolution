import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from diff_evolution.algo import (
    ConstantDE,
    ConstantSuccessRuleDE,
    RandomSuccessRuleDE,
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

linestyles = ["-", "--", "-."]
fig, ax = plt.subplots(2)

for algo_version, linestyle in zip([ConstantDE, ConstantSuccessRuleDE], linestyles):
    for retry in range(5):
        algo = algo_version()

        def call_cec(x):
            return np.sum(x ** 2)

        target_value = 0
        algo_control = AlgorithmControl(call_cec, max_fes, dims, target_value)
        best_point = algo.run(algo_control, bounds, init_population_uniform)

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
        ax[0].hlines(
            target_value, colors="red", xmin=min(x), xmax=max(x), linestyles=":",
        )
        # if algo_version != ConstantDE:
        #     ax[1].plot(
        #         list(range(len(algo.population_history))),
        #         algo.mutation_factor_history,
        #         label=f"{algo_version.__name__}-{retry} ({error})",
        #         linestyle=linestyle,
        # )
ax[0].legend()
ax[1].legend()
plt.show()

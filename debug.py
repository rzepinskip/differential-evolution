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

MAX_FES_FACTOR = 10000
TARGET_VALUE_FACTOR = 100
BOUNDS_1D = [(-100, 100)]


def plot_population_with_levels(
    algo, func_num, step=-1,
):
    points = 200
    x = np.linspace(-100, 100, points)
    y = np.linspace(-100, 100, points)
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([call_cec(x) for x in positions]).reshape(points, points)

    global_opt_raw = open(f"data/cec2017/shift_data_{func_num}.txt").readline()
    global_opt = [float(v) for v in global_opt_raw.split()]
    global_opt = global_opt[:dims]

    pop_points = list(zip(*algo.population_history[step]))
    pop_x, pop_y = pop_points[0], pop_points[1]

    fig, ax = plt.subplots(1, 1)
    multipliers = [1 * 10 ** i for i in range(0, 10)]
    m = ax.contour(
        X, Y, Z, colors="black", levels=[func_num * 100 * i for i in multipliers]
    )
    ax.clabel(m, inline=1, fontsize=8, fmt="%.2E")
    ax.scatter(pop_x, pop_y, c="blue")
    ax.scatter(global_opt[0], global_opt[1], c="red")
    plt.show()


def save_convergence_history(algo, func_num):
    fig, ax = plt.subplots()

    points = 200
    x = np.linspace(-100, 100, points)
    y = np.linspace(-100, 100, points)
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([call_cec(x) for x in positions]).reshape(points, points)

    global_opt_raw = open(f"data/cec2017/shift_data_{func_num}.txt").readline()
    global_opt = [float(v) for v in global_opt_raw.split()]
    global_opt = global_opt[:dims]

    max_iter = len(algo.population_history)

    def update_plot(i):
        ax.clear()
        pop_points = list(zip(*algo.population_history[i]))
        pop_x, pop_y = pop_points[0], pop_points[1]

        m = ax.contour(X, Y, Z, colors="black")
        ax.scatter(pop_x, pop_y, c="blue")
        ax.scatter(global_opt[0], global_opt[1], c="red")
        plt.title(f"Iteration {i} out of {max_iter}")

    anim = FuncAnimation(fig, update_plot, frames=max_iter, interval=100, repeat=False)
    anim.save(f"convergence_history_{func_num}.mp4", writer="ffmpeg")


def plot_fitness(algo, target_value):
    x = list(range(len(algo.population_history)))
    y = []
    for population in algo.population_history:
        y += [min([call_cec(member) for member in population])]

    fig, ax = plt.subplots(1, 1)
    m = ax.plot(x, y)
    ax.hlines(target_value, colors="red", xmin=min(x), xmax=max(x), linestyles="dashed")
    plt.show()


dims = 2
max_fes = MAX_FES_FACTOR * dims
bounds = BOUNDS_1D * dims

tested_funcs = range(3, 10)
for func_num in tested_funcs:
    algo = ConstantDE(seed=44)

    def call_cec(x):
        fitness = cec17_test_func(x, dims=dims, func_num=func_num)
        return fitness[0]

    target_value = TARGET_VALUE_FACTOR * func_num
    algo_control = AlgorithmControl(call_cec, max_fes, dims, target_value)
    best_point = algo.run(algo_control, bounds, init_population_uniform)

    print(f"Error [{func_num}]: {abs(call_cec(best_point) - target_value)}")

    plot_fitness(algo, target_value)
    # save_convergence_history(algo, func_num)
    # plot_population_with_levels(algo, func_num, step=0)

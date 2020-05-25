import matplotlib.pyplot as plt
import numpy as np

from diff_evolution.algo import ConstantDE, DifferentialEvolution
from diff_evolution.algo_control import AlgorithmControl
from diff_evolution.cec17_functions import cec17_test_func

MAX_FES_FACTOR = 10000
TARGET_VALUE_FACTOR = 100
BOUNDS_1D = [(-100, 100)]

func_num = 3
dims = 2

algo = ConstantDE(seed=44)


def call_cec(x):
    fitness = cec17_test_func(x, dims=dims, func_num=func_num)
    return fitness[0]


max_fes = MAX_FES_FACTOR * dims
bounds = BOUNDS_1D * dims
target_value = TARGET_VALUE_FACTOR * func_num
algo_control = AlgorithmControl(call_cec, max_fes, dims, target_value)
best_point = algo.run(algo_control, bounds)

target_value = TARGET_VALUE_FACTOR * func_num


def plot_population_with_levels(step=-1,):
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
    m = ax.contour(X, Y, Z, colors="black")
    ax.scatter(pop_x, pop_y, c="blue")
    ax.scatter(global_opt[0], global_opt[1], c="red")
    plt.show()


def plot_fitness():
    x = list(range(len(algo.population_history)))
    y = []
    for population in algo.population_history:
        y += [min([call_cec(member) for member in population])]

    fig, ax = plt.subplots(1, 1)
    m = ax.plot(x, y)
    ax.hlines(target_value, colors="red", xmin=min(x), xmax=max(x), linestyles="dashed")
    plt.show()


print(f"Error: {abs(call_cec(best_point) - target_value)}")
plot_fitness()
plot_population_with_levels(step=0)
plot_population_with_levels(step=-1)

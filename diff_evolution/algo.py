from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np

from diff_evolution.algo_control import AlgorithmControl


def init_population_uniform(population_size, bounds: List[Tuple[float, float]]):
    dimensions = len(bounds)
    pop_norm = np.random.rand(population_size, dimensions)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop_norm * diff

    return pop


# based on https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
class DifferentialEvolution(ABC):
    def __init__(
        self, crossover, seed,
    ):
        if seed:
            np.random.seed(seed)

        self.crossover = crossover

    def run(
        self, algorithm_control: AlgorithmControl, bounds: List[Tuple[float, float]], population_initializer: Callable, population_size = None
    ):
        dimensions = len(bounds)
        if not population_size:
            population_size = 4 * dimensions
        pop = population_initializer(population_size, bounds)

        fitness = np.asarray([algorithm_control.test_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        algorithm_control.update_best_fitness(fitness[best_idx])
        best = pop[best_idx]
        self.population_history = [pop]
        while algorithm_control.check_stop_criteria():
            for j in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                # mutation_factor = scaling factor
                mutation_factorant = np.clip(
                    a + self.get_mutation_factor(pop) * (b - c), [b[0] for b in bounds], [b[1] for b in bounds]
                )
                cross_points = np.random.rand(dimensions) < self.crossover
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutation_factorant, pop[j])

                f = algorithm_control.test_func(trial)
                if f is None:
                    break
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial
                        algorithm_control.update_best_fitness(f)

            self.population_history += [pop]

        return best

    @abstractmethod
    def get_mutation_factor(self, current_population):
        pass


class ConstantDE(DifferentialEvolution):
    def __init__(
        self, mutation_factor=0.8, crossover=0.9, seed=None,
    ):
        super().__init__(
            crossover=crossover, seed=seed,
        )
        self.mutation_factor = mutation_factor

    def get_mutation_factor(self, current_population):
        return self.mutation_factor


class RandomFactorDE(DifferentialEvolution):
    def __init__(
        self, crossover=0.9, seed=None,
    ):
        super().__init__(
            crossover=crossover, seed=seed,
        )

    def get_mutation_factor(self, current_population):
        return np.random.uniform(low=0.5, high=1.0)


class ConstantSuccessRuleDE(DifferentialEvolution):
    def __init__(
        self, mutation_factor=0.8, crossover=0.9, seed=None,
    ):
        super().__init__(
            crossover=crossover, seed=seed,
        )
        self.mutation_factor = mutation_factor

    def get_mutation_factor(self, current_population):
        return self.mutation_factor

    def run(
        self, algorithm_control: AlgorithmControl, bounds: List[Tuple[float, float]], population_initializer: Callable, population_size = None
    ):
        dimensions = len(bounds)
        if not population_size:
            population_size = 4 * dimensions
        pop = population_initializer(population_size, bounds)

        fitness = np.asarray([algorithm_control.test_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        algorithm_control.update_best_fitness(fitness[best_idx])
        best = pop[best_idx]
        self.population_history = [pop]
        while algorithm_control.check_stop_criteria():
            mean_prev_pop_member = np.mean(pop, axis=0)
            mean_prev_pop_member_fit = algorithm_control.test_func(mean_prev_pop_member)
            if mean_prev_pop_member_fit is None:
                break

            for j in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                mutation_factorant = np.clip(
                    a + self.get_mutation_factor(pop) * (b - c), [b[0] for b in bounds], [b[1] for b in bounds]
                )
                cross_points = np.random.rand(dimensions) < self.crossover
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutation_factorant, pop[j])

                f = algorithm_control.test_func(trial)
                if f is None:
                    break
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial
                        algorithm_control.update_best_fitness(f)

            self.population_history += [pop]

            better_than_mean = sum(fitness < mean_prev_pop_member_fit)
            if better_than_mean > 0.2 * population_size:
                self.mutation_factor = 1.22 * self.mutation_factor
            else:
                self.mutation_factor = 0.82 * self.mutation_factor
            self.mutation_factor = np.clip(self.mutation_factor, 0.5, 2.0)
        return best


class RandomSuccessRuleDE(DifferentialEvolution):
    def __init__(
        self, mutation_factor=0.8, crossover=0.9, seed=None,
    ):
        super().__init__(
            crossover=crossover, seed=seed,
        )
        self.mutation_factor = mutation_factor

    def get_mutation_factor(self, current_population):
        return self.mutation_factor

    def run(
        self, algorithm_control: AlgorithmControl, bounds: List[Tuple[float, float]], population_initializer: Callable, population_size = None
    ):
        dimensions = len(bounds)
        if not population_size:
            population_size = 4 * dimensions
        pop = population_initializer(population_size, bounds)

        fitness = np.asarray([algorithm_control.test_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        algorithm_control.update_best_fitness(fitness[best_idx])
        best = pop[best_idx]
        self.population_history = [pop]
        while algorithm_control.check_stop_criteria():
            mean_prev_pop_member = np.mean(pop, axis=0)
            mean_prev_pop_member_fit = algorithm_control.test_func(mean_prev_pop_member)
            if mean_prev_pop_member_fit is None:
                break

            for j in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                mutation_factorant = np.clip(
                    a + self.get_mutation_factor(pop) * (b - c), [b[0] for b in bounds], [b[1] for b in bounds]
                )
                cross_points = np.random.rand(dimensions) < self.crossover
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutation_factorant, pop[j])

                f = algorithm_control.test_func(trial)
                if f is None:
                    break
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial
                        algorithm_control.update_best_fitness(f)

            self.population_history += [pop]

            better_than_mean = sum(fitness < mean_prev_pop_member_fit)
            if better_than_mean > 0.2 * population_size:
                self.mutation_factor = (
                    np.random.uniform(low=1.0, high=1.5) * self.mutation_factor
                )
            else:
                self.mutation_factor = (
                    np.random.uniform(low=0.5, high=1.0) * self.mutation_factor
                )
            self.mutation_factor = np.clip(self.mutation_factor, 0.5, 2.0)

        return best

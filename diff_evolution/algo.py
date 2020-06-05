from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from diff_evolution.algo_control import AlgorithmControl


# based on https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
class DifferentialEvolution(ABC):
    def __init__(
        self, crossover, population_size, seed,
    ):
        if seed:
            np.random.seed(seed)

        self.crossover = crossover
        self.population_size = population_size

    def run(
        self, algorithm_control: AlgorithmControl, bounds: List[Tuple[float, float]]
    ):
        dimensions = len(bounds)
        pop = np.random.rand(self.population_size, dimensions)

        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([algorithm_control.test_func(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        algorithm_control.update_best_fitness(fitness[best_idx])
        best = pop_denorm[best_idx]
        self.population_history = [pop_denorm]
        while algorithm_control.check_stop_criteria():
            for j in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                # mutation_factor = scaling factor
                mutation_factorant = np.clip(
                    a + self.get_mutation_factor(pop) * (b - c), 0, 1
                )
                cross_points = np.random.rand(dimensions) < self.crossover
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutation_factorant, pop[j])
                trial_denorm = min_b + trial * diff

                f = algorithm_control.test_func(trial_denorm)
                if f is None:
                    break
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
                        algorithm_control.update_best_fitness(f)

            pop_denorm = min_b + pop * diff  # just for debug
            self.population_history += [pop_denorm]

        return best

    @abstractmethod
    def get_mutation_factor(self, current_population):
        pass


class ConstantDE(DifferentialEvolution):
    def __init__(
        self, mutation_factor=0.8, crossover=0.9, population_size=20, seed=None,
    ):
        super().__init__(
            crossover=crossover, population_size=population_size, seed=seed,
        )
        self.mutation_factor = mutation_factor

    def get_mutation_factor(self, current_population):
        return self.mutation_factor


class RandomFactorDE(DifferentialEvolution):
    def __init__(
        self, crossover=0.9, population_size=20, seed=None,
    ):
        super().__init__(
            crossover=crossover, population_size=population_size, seed=seed,
        )

    def get_mutation_factor(self, current_population):
        return np.random.uniform(low=0.5, high=1.0)


class ConstantSuccessRuleDE(DifferentialEvolution):
    def __init__(
        self, mutation_factor=0.8, crossover=0.9, population_size=20, seed=None,
    ):
        super().__init__(
            crossover=crossover, population_size=population_size, seed=seed,
        )
        self.mutation_factor = mutation_factor

    def get_mutation_factor(self, current_population):
        return self.mutation_factor

    def run(
        self, algorithm_control: AlgorithmControl, bounds: List[Tuple[float, float]]
    ):
        dimensions = len(bounds)
        pop = np.random.rand(self.population_size, dimensions)

        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([algorithm_control.test_func(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        algorithm_control.update_best_fitness(fitness[best_idx])
        best = pop_denorm[best_idx]
        self.population_history = [pop_denorm]
        while algorithm_control.check_stop_criteria():
            mean_prev_pop_member = min_b + np.mean(pop, axis=0) * diff
            mean_prev_pop_member_fit = algorithm_control.test_func(mean_prev_pop_member)

            for j in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                mutation_factorant = np.clip(
                    a + self.get_mutation_factor(pop) * (b - c), 0, 1
                )
                cross_points = np.random.rand(dimensions) < self.crossover
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutation_factorant, pop[j])
                trial_denorm = min_b + trial * diff

                f = algorithm_control.test_func(trial_denorm)
                if f is None:
                    break
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
                        algorithm_control.update_best_fitness(f)

            pop_denorm = min_b + pop * diff  # just for debug
            self.population_history += [pop_denorm]

            better_than_mean = sum(fitness < mean_prev_pop_member_fit)
            if better_than_mean > 0.2 * self.population_size:
                self.mutation_factor = 1.22 * self.mutation_factor
            else:
                self.mutation_factor = 0.82 * self.mutation_factor
            self.mutation_factor = np.clip(self.mutation_factor, 0.5, 2.0)
        return best


class RandomSuccessRuleDE(DifferentialEvolution):
    def __init__(
        self, mutation_factor=0.8, crossover=0.9, population_size=20, seed=None,
    ):
        super().__init__(
            crossover=crossover, population_size=population_size, seed=seed,
        )
        self.mutation_factor = mutation_factor

    def get_mutation_factor(self, current_population):
        return self.mutation_factor

    def run(
        self, algorithm_control: AlgorithmControl, bounds: List[Tuple[float, float]]
    ):
        dimensions = len(bounds)
        pop = np.random.rand(self.population_size, dimensions)

        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([algorithm_control.test_func(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        algorithm_control.update_best_fitness(fitness[best_idx])
        best = pop_denorm[best_idx]
        self.population_history = [pop_denorm]
        while algorithm_control.check_stop_criteria():
            mean_prev_pop_member = min_b + np.mean(pop, axis=0) * diff
            mean_prev_pop_member_fit = algorithm_control.test_func(mean_prev_pop_member)

            for j in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                mutation_factorant = np.clip(
                    a + self.get_mutation_factor(pop) * (b - c), 0, 1
                )
                cross_points = np.random.rand(dimensions) < self.crossover
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutation_factorant, pop[j])
                trial_denorm = min_b + trial * diff

                f = algorithm_control.test_func(trial_denorm)
                if f is None:
                    break
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
                        algorithm_control.update_best_fitness(f)

            pop_denorm = min_b + pop * diff  # just for debug
            self.population_history += [pop_denorm]

            better_than_mean = sum(fitness < mean_prev_pop_member_fit)
            if better_than_mean > 0.2 * self.population_size:
                self.mutation_factor = (
                    np.random.uniform(low=1.0, high=1.5) * self.mutation_factor
                )
            else:
                self.mutation_factor = (
                    np.random.uniform(low=0.5, high=1.0) * self.mutation_factor
                )
            self.mutation_factor = np.clip(self.mutation_factor, 0.5, 2.0)

        return best

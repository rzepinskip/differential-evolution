from typing import Callable, List, Tuple

EPSILON = 1e-10
RECORDING_POINTS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

class AlgorithmControl():
    def __init__(self, func: Callable[[List[float]], float], max_fes, dims, target_value):
        self.max_fes = max_fes
        self.dims = dims
        self.func = func
        self.target_value = target_value

        self.fes = 0
        self.best_fitness = None
        self.recorded_values = []

    def test_func(self, x):
        self.fes += 1
        if self.fes >= RECORDING_POINTS[len(self.recorded_values)] * self.max_fes:
            self.recorded_values.append(self.error())
        return self.func(x)

    def error(self):
        if self.best_fitness is None:
            return None
        return self.best_fitness - self.target_value

    def check_stop_criteria(self):
        if self.fes >= self.max_fes or (self.best_fitness is not None and abs(self.error()) < EPSILON):
            return False
        return True

    def update_best_fitness(self, fitness):
        self.best_fitness = fitness

    def fill_up_recorder_values(self):
        self.recorded_values.extend([self.error()] * (len(RECORDING_POINTS) - len(self.recorded_values)))
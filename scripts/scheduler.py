import numpy as np
class GammaScheduler:
    def __init__(self, min, steps) -> None:
        self.s = 0
        self.steps = steps
        self.min = min
    
    def step(self) -> int:
        self.s += 1
        if self.s < self.steps:
            return (1 - self.min) * 0.5 * (1 + np.cos(np.pi * self.s / self.steps)) + self.min
        else:
            return self.min
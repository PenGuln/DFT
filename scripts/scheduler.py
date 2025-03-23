import numpy as np
class GammaScheduler:
    def __init__(self, min, steps, gc) -> None:
        self.s = 0
        self.steps = steps
        self.min = min
        self.gc = gc
    
    def step(self) -> int:
        self.s += 1
        t = self.s // self.gc
        if t < self.steps:
            return (1 - self.min) * 0.5 * (1 + np.cos(np.pi * t / self.steps)) + self.min
        else:
            return self.min

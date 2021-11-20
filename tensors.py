import numpy as np


class tensor:
    def __init__(self, shape):
        self.elems = np.ndarray(shape, np.float64)
        self.grads = np.ndarray(shape, np.float64)

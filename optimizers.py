import numpy as np


class optimizer(object):
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError

    def zero_grads(self):
        for param in self.params:
            param.grads = 0


class sgd(optimizer):
    def __init__(self, params, lr, wd, moment):
        super().__init__(params)
        self.lr = lr
        self.wd = wd
        self.moment = moment
        self.vel = []
        for param in params:
            self.vel.append(np.zeros_like(param.grads))

    def step(self):
        for param, vel in zip(self.params, self.vel):
            vel = self.moment * vel + param.grads + self.wd * param.elems
            param.elems = param.elems - self.lr * vel

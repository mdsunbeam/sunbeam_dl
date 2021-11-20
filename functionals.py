import numpy as np
from tensors import tensor


class functional(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def params(self):
        return []


class linear(functional):
    def __init__(self, input, output):
        self.w = tensor((input, output))
        self.b = tensor((1, output))

    def forward(self, x):
        output = np.dot(x, self.w.elems) + self.b.elems
        self.input = x
        return output

    def backward(self, dy):
        self.w.grads += np.dot(self.input.T, dy)
        self.b.grads += np.sum(dy, axis=0, keepdims=True)
        grads_input = np.dot(dy, self.w.elems.T)
        return grads_input

    def params(self):
        return [self.w, self.b]

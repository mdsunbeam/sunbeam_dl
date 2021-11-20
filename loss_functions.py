class loss_function(object):
    def __init__(self):
        raise NotImplementedError

    def __loss__(self):
        raise NotImplementedError


class mse(loss_function):
    def __init__(self, input, target):
        super().__init__()
        self.input = input
        self.target = target
        self.mse = 1 / (target.size) * (input - target) ** 2

    def __loss__(self):
        return self.mse

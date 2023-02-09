import numpy as np

from optimizer import Optimizer

class RegularizedSGD(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 alpha: float = 0.1) -> None:
        super().__init__()
        self.lr = lr
        self.alpha = alpha

    def step(self) -> None:

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:

            # Use this to update parameters
            kwargs['param'] -= (
                self.lr * kwargs['grad'] + self.alpha * kwargs['param'])
import numpy as np

from param_operation import ParamOperation

class BiasAdd(ParamOperation):

    def __init__(self,
                 B: np.ndarray):
        super().__init__(B)

    def _output(self, inference: bool) -> np.ndarray:
        return self.input_ + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        output_grad_reshape = np.sum(output_grad, axis=0).reshape(1, -1)
        param_grad = np.ones_like(self.param)
        return param_grad * output_grad_reshape

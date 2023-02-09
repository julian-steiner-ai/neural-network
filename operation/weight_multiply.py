import numpy as np

from param_operation import ParamOperation

class WeightMultiply(ParamOperation):
    def __init__(self, W: np.ndarray):
        super().__init__(W)

    def _output(self, inference: bool) -> np.ndarray:
        return np.matmul(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.matmul(output_grad, self.param.transpose(1, 0))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.matmul(self.input_.transpose(1, 0), output_grad)

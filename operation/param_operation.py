import numpy as np

from operation.operation import Operation
from utils.utils import assert_same_shape

class ParamOperation(Operation):

    def __init__(self, param: np.ndarray) -> np.ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

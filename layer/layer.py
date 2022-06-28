"""
"""

from typing import List
import numpy as np
from ..operation import Operation, ParamOperation
from ..utils import assert_same_shape

class Layer(object):
    """
    """
    def __init__(self,
                 neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: np.ndarray) -> None:
        pass

    def forward(self, input_: np.ndarray, inference=False) -> np.ndarray:
        """
        """
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        """
        assert_same_shape(self.output, output_grad)

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert_same_shape(self.input_, input_grad)

        self._param_grads()

        return input_grad

    def _param_grads(self) -> None:
        """
        """
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:
        """
        """
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

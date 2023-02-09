import numpy as np

from operation import Operation


class Sigmoid(Operation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return 
import numpy as np

from operation.operation import Operation

class Linear(Operation):
    '''
    Linear activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad
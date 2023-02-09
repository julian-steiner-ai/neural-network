import numpy as np

from layer.layer import Layer

class Conv1D(Layer):
    """
    Convolution 1D Layer.
    """
    def __init__(self, neurons: int):
        super().__init__(neurons)

    def forward(self, _input: np.ndarray) -> np.ndarray:
        return super().forward(_input)
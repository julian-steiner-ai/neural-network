"""
Implementation of mean squared error loss function.
"""
import numpy as np
from .loss import Loss

class MeanSquaredError(Loss):
    """
    Class for MeanSquaredError
    """
    def __init__(self) -> None:
        """
        Pass
        """
        super().__init__()

    def _output(self) -> float:
        """
        Computes the per-observation squared error loss
        """
        loss = (
            np.sum(np.power(self.prediction - self.target, 2)) / 
            self.prediction.shape[0]
        )

        return loss

    def _input_grad(self) -> np.ndarray:
        """
        Computes the loss gradient with respect to the input for MSE loss
        """        

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]

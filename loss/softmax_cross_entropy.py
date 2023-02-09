import numpy as np

from loss import Loss
from utils import normalize, softmax, unnormalize

class SoftmaxCrossEntropy(Loss):
    def __init__(self, eps: float=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:
        if self.target.shape[1] == 0:
            self.single_class = True

        if self.single_class:
            self.prediction, self.target = \
            normalize(self.prediction), normalize(self.target)

        softmax_preds = softmax(self.prediction, axis=1)

        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        softmax_cross_entropy_loss = (
            -1.0 * self.target * np.log(self.softmax_preds) - \
                (1.0 - self.target) * np.log(1 - self.softmax_preds)
        )

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> np.ndarray:
        if self.single_class:
            return unnormalize(self.softmax_preds - self.target)
        else:
            return (self.softmax_preds - self.target) / self.prediction.shape[0]
import numpy as np

from layer.layer import Layer

from operation.activation.linear import Linear

from operation.operation import Operation
from operation.dropout import Dropout

from operation.cnn.conv2D import Conv2D as Conv2DOperation
from operation.cnn.flatten import Flatten

class Conv2D(Layer):
    """
    Convolutional Layer for 2D ndarrays
    """
    def __init__(self,
                 out_channels: int,
                 param_size: int,
                 dropout: int = 1.0,
                 weight_init: str = "normal",
                 activation: Operation = Linear(),
                 flatten: bool = False) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.dropout = dropout
        self.weight_init = weight_init
        self.out_channels = out_channels
    
    def _setup_layer(self, input_: np.ndarray) -> None:
        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2/(in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(loc=0,
                                      scale=scale,
                                      size=(input_.shape[1],  # input channels
                                      self.out_channels,
                                      self.param_size,
                                      self.param_size))

        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2DOperation(conv_param))
        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(Flatten())

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None
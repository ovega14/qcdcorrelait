import torch
import torch.nn.functional as F

from typing import List, Optional, Union, TypeVar
TorchFunctional = TypeVar('TorchFunctional')


class MLP(torch.nn.Module):
    """
    Basic multilayer perceptron. Uses a combination of linear and batch norm 
    layers.

    Args:
        input_dim: Number of features in input data
        output_dim: Number of features in output data
        hidden_dims: List of dimensions of data output by hidden layers
        batch_norm: Whether to use batch normalization (default False)
        activation: Activation function used by the network (default F.elu)

    Attributes:
        depth: number of layers (including output layer) in the network
        fc_layers: ModuleList of fully connected layers
        bn_layers: ModuleList of batch norm layers; identity if 
            `batch_norm = False`
        activation: Activation function
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        batch_norm: Optional[bool] = False,
        activation: Optional[TorchFunctional] = F.elu
    ):
        super().__init__()

        assert len(hidden_dims) > 0, \
            'Must specify dimensions for at least one hidden layer.'
        self.depth = len(hidden_dims) + 1

        # Fully connected (linear) layers
        fc_layers = [torch.nn.Linear(input_dim, hidden_dims[0])]
        for i in range(1, self.depth - 1):
            fc_layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        fc_layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        self.fc_layers = torch.nn.ModuleList(fc_layers)

        # (Optional) batch normalization layers
        if batch_norm:
            bn_layers = []
            for i in range(self.depth - 1):
                bn_layers.append(torch.nn.BatchNorm1d(hidden_dims[i]))
            bn_layers.append(torch.nn.BatchNorm1d(output_dim))
        else:
            bn_layers = [torch.nn.Identity()] * self.depth
        self.bn_layers = torch.nn.ModuleList(bn_layers)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feeds data through neural net, using residual connections."""
        temp = x
        for i in range(self.depth):
            temp = self.fc_layers[i](temp)
            temp = self.bn_layers[i](temp)
            temp = self.activation(temp)
        x = temp + x
        return x
    

class CNN(torch.nn.Module):
    """
    Basic convolutional neural network. Uses a combination of 1d convolutional 
    and batch normalization layers

    Args:
        in_channels: Number of channels in input data
        out_channels: Number of channels in output data
        hidden_channels: List of number of channels in data output from each 
            hidden layer
        kernel_size: Size of the convolving kernel
        batch_norm: Whether to use batch normalization
        activation: Activation function used in the network

    Attributes:
        depth: Number of hiddden layers (including output layer) in the network
        conv_layers: ModuleList of convolutional layers
        bn_layers: (Optional) ModuleList of batch normalization layers
        activation: The activation function being used at each hidden layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        kernel_size: int,
        batch_norm: Optional[bool] = False,
        activation: Optional[TorchFunctional] = F.elu
    ):
        super().__init__()

        self.depth = len(hidden_channels) + 1

        # Convolutional layers
        conv_layers = [torch.nn.Conv1d(
                            in_channels = in_channels, 
                            out_channels = hidden_channels[0],
                            kernel_size = kernel_size,
                            padding = 'same'
                        )]
        for i in range(1, self.depth - 1):
            conv_layers.append(torch.nn.Conv1d(
                                    in_channels = hidden_channels[i-1],
                                    out_channels = hidden_channels[i],
                                    kernel_size = kernel_size,
                                    padding = 'same'
                                ))
        conv_layers.append(torch.nn.Conv1d(
                                in_channels = hidden_channels[-1], 
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                padding = 'same'
                            ))
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        
        # (Optional) batch normalization layers
        if batch_norm:
            bn_layers = []
            for i in range(self.depth - 1):
                bn_layers.append(torch.nn.BatchNorm1d(hidden_channels[i]))
            bn_layers.append(torch.nn.BatchNorm1d(out_channels))
        else:
            bn_layers = [torch.nn.Identity()] * self.depth 
        self.bn_layers = torch.nn.ModuleList(bn_layers)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feeds data through network and uses residual connections. For input to 
        CNN, the data is assumed to not possess a `channels` dimension, so the 
        data is unsqueezed at dimension 1 and then squeezed again at output.        
        """
        x = torch.unsqueeze(x, dim=1)
        
        temp = x
        for i in range(self.depth):
            temp = self.conv_layers[i](temp)
            temp = self.bn_layers[i](temp)
            temp = self.activation(temp)
        x = temp + x
        
        x = torch.squeeze(x, dim=1)
        return x
    

class LinearModel(torch.nn.Module):
    """
    A basic fully linear model with a single layer.

    Args:
        input_dim: Number of features in input data
        output_dim: Number of features in output_data

    Attributes:
        layers: ModuleList of linear layers
        depth: Number of linear layers in the network
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim)
        ])
        self.depth = len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = x
        for i in range(self.depth):
            temp = self.layers[i](temp)
        x = temp
        return x
    

class Transformer(torch.nn.Module):
    """
    Basic transformer model.

    Args:
        input_dim: Number of features in the input data
        num_heads: Number of heads to use for multihead attention
        dim_feedforward: Dimension of feedforward in transformer encoder layers
        batch_first: Whether batch dimension is first in data
        activation: Activation function to use in transformer encoder layers
    
    Attributes:
        depth: Number of transformer encoder layers
        layers: ModuleList of transformer encoder layers
    """
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        dim_feedforward: Union[bool, int] = None,
        batch_first: Optional[bool] = True,
        activation: Optional[str] = 'gelu'
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward: int = 4 * input_dim
        
        self.depth = 2
        
        self.layers = torch.nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(
                torch.nn.TransformerEncoderLayer(
                    d_model = input_dim,
                    nhead = num_heads,
                    dim_feedforward = dim_feedforward,
                    batch_first = batch_first,
                    activation = activation
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feeds data through the network using residual connections."""
        x = torch.unsqueeze(x, -1)

        temp = x
        for i in range(self.depth):
            temp = self.layers[i](temp)
            temp = F.elu(temp)
        x = temp + x

        x = torch.squeeze(x, -1)
        return x

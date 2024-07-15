import torch
import torch.nn.functional as F

from typing import List, Optional


class MLP(torch.nn.Module):
    """
    Basic multilayer perceptron. Uses a combination of linear, batch norm, and activation layers.

    Args:
        input_dim: Number of features in input data
        output_dim: Number of features in output data
        hidden_dims: List of dimensions of data output by hidden layers
        batch_norm: Whether to use batch normalization (default False)
        activation: Activation function used by the network (default F.elu)

    Attributes:
        depth: number of layers (including output layer) in the network
        fc_layers: ModuleList of fully connected layers
        bn_layers: ModuleList of batch norm layers; identity if `batch_norm = False`
        activation: Activation function
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        batch_norm: Optional[bool] = False,
        activation: Optional[torch.nn.Functional] = F.elu
    ):
        super().__init__()

        assert len(hidden_dims) > 0, 'Must specify dimensions for at least one hidden layer.'
        self.depth = len(hidden_dims) + 1

        # Fully connected (linear) layers
        fc_layers = [torch.nn.Linear(input_dim, hidden_dims[0])]
        for i in range(1, self.depth - 1):
            fc_layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        fc_layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        self.fc_layers = torch.nn.ModuleList(*fc_layers)

        # (Optional) batch normalization layers
        if batch_norm:
            bn_layers = [torch.nn.BatchNorm1d(hidden_dims[i]) for i in range(self.depth - 1)]
            bn_layers.append(torch.nn.BatchNorm1d(output_dim))
        else:
            bn_layers = [torch.nn.Identity()] * self.depth
        bn_layers = torch.nn.ModuleList(*bn_layers)

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
    Basic convolutional neural network.

    Args:
        in_channels: Number of channels in input data
        out_channels: Number of channels in output data
        hidden_channels: List of number of channels in data output from each hidden layer
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
        out_chanels: int,
        hidden_channels: List[int],
        kernel_size: int,
        batch_norm: Optional[bool] = False,
        activation: Optional[torch.nn.Functional] = F.elu
    ):
        super().__init__()

        self.depth = len(hidden_channels) + 1

        # Convolutional layers
        conv_layers = [torch.nn.Conv1d(
                            in_channels=in_channels, 
                            out_chanels=hidden_channels[0],
                            kernel_size=kernel_size
                        )]
        for i in range(1, self.depth - 1):
            conv_layers.append(torch.nn.Conv1d(
                                    in_channels=hidden_channels[i-1],
                                    out_chanels=hidden_channels[i],
                                    kernel_size=kernel_size
                                ))
        conv_layers.append(torch.nn.Conv1d(
                                in_channels=hidden_channels[-1], 
                                out_channels=out_chanels,
                                kernel_size=kernel_size
                            ))
        self.conv_layers = torch.nn.ModuleList(*conv_layers)
        
        # (Optional) batch normalization layers
        if batch_norm:
            bn_layers = [torch.nn.BatchNorm1d(hidden_channels[i]) for i in range(self.depth - 1)]
            bn_layers.append(torch.nn.BatchNorm1d(out_chanels))
        else:
            bn_layers = [torch.nn.Identity()] * self.depth 
        bn_layers = torch.nn.ModuleList(*bn_layers)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feeds data through network and uses residual connections. For input to CNN, the data is
        assumed to not possess a `channels` dimension, so the data is unsqueezed at dimension 1 and
        then squeezed again at the output.        
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

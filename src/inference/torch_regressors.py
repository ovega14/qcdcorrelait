import torch
import torch.nn.functional as F

from typing import List, Optional


class MLP(torch.nn.Module):
    """
    Simple Multilayer Perceptron.

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

        # (Optional) batch normalization
        if batch_norm:
            bn_layers = [torch.nn.BatchNorm1d(hidden_dims[i]) for i in range(self.depth - 1)]
            bn_layers.append(torch.nn.BatchNorm1d(output_dim))
        else:
            bn_layers = [torch.nn.Identity()] * (self.depth - 1)
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

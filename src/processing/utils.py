import torch
import numpy as np

import numpy.typing as npt
from typing import Tuple


#===================================================================================================
# ARRAY / TENSOR OPERATIONS
#===================================================================================================
def tensor_means_stds_by_axis0(tensor: torch.Tensor) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the means and standard deviations of a 2d torch tensor by `axis=0`.

    Args:
        tensor: 2d torch.Tensor, the tensor to be averaged.
        
    Return:
        means: 1d numpy.array
        stds: 1d numpy.array
    """
    means, stds = [], []
    for i in range(tensor.shape[1]):
        means.append(np.average(tensor.numpy()[:, i]))
        stds.append(np.std(tensor.numpy()[:, i]))
    means = np.array(means)
    stds = np.array(stds)
    return means, stds

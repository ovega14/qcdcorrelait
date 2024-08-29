"""Utilities for normalizing and standardizing correlator data."""
import torch
import numpy as np

import numpy.typing as npt
from typing import Tuple


# =============================================================================
#  N-DIMENSIONAL NORMALIZATION
# =============================================================================
def normalize_1d(corr: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Normalize a 1d correlator.

    Convention:
    corr = normalized_corr * normalzation_factors

    Args:
        corr: 2d numpy.array, correlator data. Order as (t, other).
            The dimension ``other`` could be conf or tsrc. This function itself
            does not need to specify the physical meaning of data dimensions.

    Returns:
        normalized_corr: 1d numpy.array, normalized correlator. Order as (t).
        normalzation_factors: 2d numpy.array, normalization factors. Order as  
            (t, other).
    """
    normalzation_factors = 1. / np.mean(corr, axis=0)
    normalized_corr = corr / np.mean(corr, axis=0)

    return normalized_corr, normalzation_factors


def normalize_2d(corr: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Divide correlator samples by their average over dimensions other than t 
    for a 2d numpy.array correlator

    Convention:
        corr = normalized_corr * normalization_factors

    Args:
        corr: 2d numpy.array, correlator data. Order as (t, other).
            The dimension ``other`` could be conf or tsrc. This function itself 
            does not need to specify the physical meaning of data dimensions.

    Returns:
        normalized_corr: 1d numpy.array, normalized correlator. Order as (t).
        normalzation_factors: 2d numpy.array, normalization factors. Order as 
            (t, other).
    """
    normalzation_factors = 1. / np.mean(corr, axis=(1)).reshape((-1, 1))
    normalized_corr = corr / np.mean(corr, axis=(1)).reshape(-1, 1)

    return normalized_corr, normalzation_factors


def normalize_3d(corr: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Divide correlator samples by their average over conf, tsrc for a 3d 
    numpy.NDArray correlator.

    Convention:
        corr = normalized_corr * normalzation_factors

    Args:
        corr: 3d `numpy.array`, correlator data. Order as (t, conf, tsrc).

    Returns:
        normalized_corr: 1d numpy.array, normalized correlator. Order as (t,).
        normalzation_factors: 3d numpy.array, normalization factors. Order as 
            (t, conf, tsrc).
    """
    normalzation_factors = 1. / np.mean(corr, axis=(1, 2)).reshape((-1, 1, 1))
    normalized_corr = corr / np.mean(corr, axis=(1, 2)).reshape(-1, 1, 1)

    return normalized_corr, normalzation_factors


# =============================================================================
#  AVERAGING
# =============================================================================
def average_tsrc(corr: npt.NDArray) -> npt.NDArray:
    """
    Averages correlator data over the source times and orders by configuration.

    Args:
        numpy array of shape [Nt, Nconf, Ntsrc]
    
    Returns:
        numpy array of shape [Nconf, Nt]
    """
    return np.average(corr, axis=2).transpose()  # [nconf, tau]


# =============================================================================
#  STANDARDIZATION
# =============================================================================
def tensor_means_stds_by_axis0(
    tensor: torch.Tensor
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute means and standard deviations of a 2d torch tensor by `axis=0`.

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

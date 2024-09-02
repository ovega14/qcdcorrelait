"""Utilities for indexing and converting correlators between data types."""
import torch
import numpy as np
import gvar as gv

import numpy.typing as npt
from typing import Optional, TypeVar
GVDataset = TypeVar('GVDataset')


# =============================================================================
#  INDEXING AND CONVERTING
# =============================================================================
def tensor_data_by_ind_list(
    raw_corr: npt.NDArray, 
    ind_list: Optional[list[int]] = None
) -> torch.Tensor:
    """
    Converts a 3d numpy array `raw_corr` to a 2d torch tensor (partially 
    flattened according to `ind_list`).

    Args:
        ind_list: List of integer indices for the time sources
        raw_corr: 3d numpy array of shape [num_tau, num_cfgs, num_tsrc]

    Returns:
        Reshaped PyTorch tensor of `raw_corr` with source times filtered by 
        `ind_list`, shaped as `[num_cfgs * len(ind_list), num_tau]`.
    """
    if ind_list is None:
        ind_list = list(range(raw_corr.shape[-1]))
    raw_corr = torch.from_numpy(raw_corr[:, :, ind_list]).double()
    return torch.flatten(raw_corr, start_dim=1, end_dim=-1).T


def tensor_data_to_np_data_3d(
    tensor_corr: torch.Tensor,
    num_cfgs: int,
    num_taus: int
) -> npt.NDArray:
    """
    Converts correlator data from a PyTorch tensor to a 3d numpy array.

    Args:
        tensor_corr: torch.Tensor shaped [num_cfgs * len(ind_list), num_tau]
        num_configs: number of configurations in data

    Returns:
        reshaped numpy 3d array; [n_taus, n_configs, len(ind_list)]
    """
    np_corr = tensor_corr.T.numpy().reshape((num_taus, num_cfgs, -1))
    return np_corr


def convert_to_gvars(
    dict_orig_corrs: dict[str, npt.NDArray], 
    averages_tsrc: Optional[bool] = False
) -> GVDataset:
    """
    Converts correlators in the form `[Nt, nconf, Ntsrc]` to a `gvar` dataset.

    Args:
        dict_orig_corrs: dict of correlators in the original format
            keys: names - list of strings, giving key names to corr in corrs
            values: corrs - list of np arrays, each shaped `[Nt, nconf, Ntsrc]`
        averages_tsrc (bool): averages over the time sources (3d arrays of 
            corrrelators) or not (2d arrays of correlators)

    Returns:
        Correlated gvar dataset with correlators referenced by names
    """
    corr_dict = {}
    for name, corr in dict_orig_corrs.items():
        if averages_tsrc:
            # corr_dict[name] = average_tsrc(corr)
            corr_dict[name] = np.average(corr, axis=-1).transpose()
        else:
            corr_dict[name] = corr.transpose()
    return gv.dataset.avg_data(corr_dict)  # (n_cf, n_tau)


def tensor_to_avg_over_tsrc(
    tensor: torch.Tensor, 
    n_tau: int, 
    n_configs: int
) -> npt.NDArray:
    """
    Converts a torch.Tensor into a numpy array and averages over the source 
    times axis.
    
    Args:
        tensor: A `torch.tensor` shaped `[num_cfgs * len(ind_list), num_tau]`
        
    Returns:
        A 1d numpy array of shape [num_tau] that has been averaged over 
        source times
    """
    temp = tensor.T.reshape((n_tau, n_configs, -1)).numpy()
    return np.average(temp, axis=-1)

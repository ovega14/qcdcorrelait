import h5py
import numpy as np
import gvar as gv
import torch

import numpy.typing as npt
from typing import (
    List, Tuple, Optional, TypeVar, Union
)
GVDataset = TypeVar('GVDataset')

from .utils import tensor_means_stds_by_axis0


"""Utilities for processing correlator data."""
NUM_TSRC: int = 24  # 24 source times for our datasets.


def get_corrs(h5fname: str, corr_tags: List[str]) -> List[npt.NDArray]:
    """
    Retrieve correlator data from hdf5 cache. Order as (t, conf, tsrc).

    Args:
        h5fname: str, name/location of the hdf5 cache.
        corr_tags: correlator names

    Returns:
        List of reshaped correlators corresponding to corr_tags.
    """
    data = h5py.File(h5fname)['data']
    global NUM_TSRC
    
    result = []
    for tag in corr_tags:
        corr_flat = data[tag]
        np_container = np.zeros(corr_flat.shape)
        corr_flat.read_direct(np_container)
        corr = np_container.transpose().reshape((corr_flat.shape[-1], -1, NUM_TSRC))
        result.append(corr)
    return result


def preprocess_data(
    corr_i: npt.NDArray,
    corr_o: npt.NDArray,
    train_ind_list: List[int],
    bc_ind_list: List[int],
    unlab_ind_list: List[int]
) -> dict[str, npt.NDArray]:
    """
    Preprocesses the correlator dataset into a dictionary of input/output data, means, and stdevs.

    Args:
        corr_i: Input correlator data, 3d numpy array
        corr_o: Output correlator data, 3d numpy array
        train_ind_list: Source time indices for training
        bc_ind_list: Source time indices for bias-correction
        unlab_ind_list: Source time indices for unlabeled data.

    Returns:
        dict_data: Dictionary of preprocessed data.
    """
    corr_i_tensor = tensor_data_by_ind_list(corr_i)
    corr_o_tensor = tensor_data_by_ind_list(corr_o)

    corr_i_train_tensor = tensor_data_by_ind_list(corr_i, ind_list=train_ind_list)
    corr_o_train_tensor = tensor_data_by_ind_list(corr_o, ind_list=train_ind_list)

    corr_i_bc_tensor = tensor_data_by_ind_list(corr_i, ind_list=bc_ind_list)
    corr_o_bc_tensor = tensor_data_by_ind_list(corr_o, ind_list=bc_ind_list)

    corr_i_unlab_tensor = tensor_data_by_ind_list(corr_i, ind_list=unlab_ind_list)
    corr_o_unlab_tensor = tensor_data_by_ind_list(corr_o, ind_list=unlab_ind_list)

    # Compute means and stds of training data for normalization & denormalization
    corr_i_train_means, corr_i_train_stds = tensor_means_stds_by_axis0(corr_i_train_tensor)
    corr_o_train_means, corr_o_train_stds = tensor_means_stds_by_axis0(corr_o_train_tensor)

    dict_data: dict[str, np.NDArray] = dict()
    dict_data["corr_i_tensor"] = corr_i_tensor
    dict_data["corr_o_tensor"] = corr_o_tensor
    dict_data["corr_i_train_tensor"] = corr_i_train_tensor
    dict_data["corr_o_train_tensor"] = corr_o_train_tensor
    dict_data["corr_i_bc_tensor"] = corr_i_bc_tensor
    dict_data["corr_o_bc_tensor"] = corr_o_bc_tensor
    dict_data["corr_i_unlab_tensor"] = corr_i_unlab_tensor
    dict_data["corr_o_unlab_tensor"] = corr_o_unlab_tensor
    dict_data["corr_i_train_means"] = corr_i_train_means
    dict_data["corr_i_train_stds"] = corr_i_train_stds
    dict_data["corr_o_train_means"] = corr_o_train_means
    dict_data["corr_o_train_stds"] = corr_o_train_stds
    
    return dict_data


#===================================================================================================
# DATA NORMALIZATION
#===================================================================================================
def normalize_3d(corr: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Divide correlator samples by their average over conf, tsrc for a 3d numpy.NDArray correlator.

    Convention:
        corr = normalized_corr * normalzation_factors

    Args:
        corr: 3d numpy.array, correlator data. Order as (t, conf, tsrc).

    Returns:
        normalized_corr: 1d numpy.array, normalized correlator. Order as (t,).
        normalzation_factors: 3d numpy.array, normalization factors. Order as (t, conf, tsrc).
    """
    normalzation_factors = 1. / np.mean(corr, axis=(1, 2)).reshape((-1, 1, 1))
    normalized_corr = corr / np.mean(corr, axis=(1, 2)).reshape(-1, 1, 1)

    return normalized_corr, normalzation_factors


def normalize_2d(corr: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Divide correlator samples by their average over the dimensions other than t.
    for a 2d numpy.array correlator

    Convention:
        corr = normalized_corr * normalzation_factors

    Args:
        corr: 2d numpy.array, correlator data. Order as (t, other).
        The dimension ``other`` could be conf or tsrc.
        This function itself does not need to specify the physical meaning of data dimensions.

    Returns:
        normalized_corr: 1d numpy.array, normalized correlator. Order as (t).
        normalzation_factors: 2d numpy.array, normalization factors. Order as (t, other).
    """
    normalzation_factors = 1. / np.mean(corr, axis=(1)).reshape((-1, 1))
    normalized_corr = corr / np.mean(corr, axis=(1)).reshape(-1, 1)

    return normalized_corr, normalzation_factors


def normalize_1d(corr: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Normalize a 1d correlator.

    Convention:
    corr = normalized_corr * normalzation_factors

    Args:
        corr: 2d numpy.array, correlator data. Order as (t, other).
        The dimension ``other`` could be conf or tsrc.
        This function itself does not need to specify the physical meaning of data dimensions.

    Returns:
        normalized_corr: 1d numpy.array, normalized correlator. Order as (t).
        normalzation_factors: 2d numpy.array, normalization factors. Order as (t, other).
    """
    normalzation_factors = 1. / np.mean(corr, axis=0)
    normalized_corr = corr / np.mean(corr, axis=0)

    return normalized_corr, normalzation_factors


def average_tsrc(corr: npt.NDArray) -> npt.NDArray:
    """
    Averages correlator data over the source times and orders by configuration.

    Args:
        numpy array of shape [Nt, Nconf, Ntsrc]
    Returns:
        numpy array of shape [Nconf, Nt]
    """
    return np.average(corr, axis=2).transpose()  # [nconf, tau]


#===================================================================================================
# MODEL INPUT PREPARATION
#===================================================================================================
def prepare_input(
    corr_s: npt.NDArray,
    corr_l: npt.NDArray,
    train_ind_list: List[int],
    bc_ind_list: List[int],
    unlab_ind_list: List[int],
    t_extent: int
) -> Tuple[npt.NDArray, ...]:
    """
    Prepares input data from normalized strange and light correlators for supervised learning.

    The major and secondary axes for "axis-0 of X" are configurations and source times, 
    respectively. Similarly for "axis-0 of Y."

    Args:
        corr_s:
        corr_l:
        train_ind_list: List of source time indices for training data
        bc_ind_list: List of source time indices for bias-correction data
        unlab_ind_list: List of source time indices for unlabeled data
        t_extent: The time extent
    
    Returns:
        X_train: numpy NDArray shaped [n_train * num_configs, num_t_extents]
        Y_train: numpy NDArray shaped [n_train * num_configs, ]
        X_bc: numpy NDArray shaped [n_bc * num_configs, num_t_extents]
        Y_bc: numpy NDArray shaped [n_bc * num_configs, ]
        X_unlab: numpy NDArray shaped [n_unlab * num_configs, num_t_extents]
        Y_unlab:
        norm_factors_X:
        norm_factors_Y:
    """
    num_tau = corr_s.shape[0]

    # Retrieve and normalize strange correlator data
    X_train = corr_s[:, :, train_ind_list]
    X_bc = corr_s[:, :, bc_ind_list]
    X_unlab = corr_s[:, :, unlab_ind_list]

    X_train, norm_factors_X = normalize_3d(X_train)
    X_bc = X_bc * norm_factors_X
    X_unlab = X_unlab * norm_factors_X

    # Retrieve and normalize (output) light correlator data
    Y_train = corr_l[t_extent, :, train_ind_list]
    Y_bc = corr_l[t_extent, :, bc_ind_list]
    Y_unlab = corr_l[t_extent, :, unlab_ind_list]

    norm_factors_Y = 1. / np.average(Y_train)
    
    Y_train = Y_train * norm_factors_Y
    Y_bc = Y_bc * norm_factors_Y
    Y_unlab = Y_unlab * norm_factors_Y

    # Reshape all data
    X_train = X_train.reshape((num_tau, -1)).transpose()
    X_bc = X_bc.reshape((num_tau, -1)).transpose()
    X_unlab = X_unlab.reshape((num_tau, -1)).transpose()

    Y_train = Y_train.flatten()
    Y_bc = Y_bc.flatten()
    Y_unlab = Y_unlab.flatten()

    norm_factors_X = norm_factors_X.flatten()  # [num_train * num_cfgs * num_tau]

    return X_train, Y_train, X_bc, Y_bc, X_unlab, Y_unlab, norm_factors_X, norm_factors_Y


def prepare_input_seq2seq(
    corr_s: npt.NDArray,
    corr_l: npt.NDArray,
    train_ind_list: List[int],
    bc_ind_list: List[int],
    unlab_ind_list: List[int]
) -> Tuple[npt.NDArray, ...]:
    """
    Prepares input data from normalized strange and light correlators for sequence to sequence
    supervised learning.

    The major and secondary axes for "axis-0 of X" are configurations and source times, 
    respectively. Similarly for "axis-0 of Y."

    Args:
        corr_s:
        corr_l:
        train_ind_list: List of source time indices for training data
        bc_ind_list: List of source time indices for bias-correction data
        unlab_ind_list: List of source time indices for unlabeled data
        t_extent: The time extent
    
    Returns:
        X_train: numpy NDArray shaped [n_train * num_configs, num_t_extents]
        Y_train: numpy NDArray shaped [n_train * num_configs, ]
        X_bc: numpy NDArray shaped [n_bc * num_configs, num_t_extents]
        Y_bc: numpy NDArray shaped [n_bc * num_configs, ]
        X_unlab: numpy NDArray shaped [n_unlab * num_configs, num_t_extents]
        Y_unlab:
        norm_factors_X:
        norm_factors_Y:
    """
    num_tau = corr_s.shape[0]

    X_train = corr_s[:, :, train_ind_list]
    X_bc = corr_s[:, :, bc_ind_list]
    X_unlab = corr_s[:, :, unlab_ind_list]

    # normalization
    X_train, norm_factors_X = normalize_3d(X_train)
    X_bc *= norm_factors_X
    X_unlab *= norm_factors_X

    Y_train = corr_l[:, :, train_ind_list]
    Y_bc = corr_l[:, :, bc_ind_list]
    Y_unlab = corr_l[:, :, unlab_ind_list]

    Y_train, norm_factors_Y = normalize_3d(Y_train)
    Y_bc *= norm_factors_Y
    Y_unlab *= norm_factors_Y

    # reshaping
    X_train = X_train.reshape((num_tau, -1)).transpose()
    X_bc = X_bc.reshape((num_tau, -1)).transpose()
    X_unlab = X_unlab.reshape((num_tau, -1)).transpose()

    Y_train = Y_train.reshape((num_tau, -1)).transpose()
    Y_bc = Y_bc.reshape((num_tau, -1)).transpose()
    Y_unlab = Y_unlab.reshape((num_tau, -1)).transpose()

    norm_factors_X = norm_factors_X.flatten()
    norm_factors_Y = norm_factors_Y.flatten()

    return X_train, Y_train, X_bc, Y_bc, X_unlab, Y_unlab, norm_factors_X, norm_factors_Y


#===================================================================================================
# INDEXING AND CONVERTING
#===================================================================================================
def tensor_data_by_ind_list(
    raw_corr: npt.NDArray, 
    ind_list: Optional[List[int]] = list(range(NUM_TSRC))
) -> torch.Tensor:
    """
    Convets a 3d numpy array `raw_corr` to a 2d torch tensor (partially flattened according to
    `ind_list`).

    Args:
        ind_list: List of integer indices for the time sources
        raw_corr: 3d numpy array of shape [num_tau, num_cfgs, num_tsrc]

    Returns:
        Reshaped PyTorch tensor of `raw_corr` with source times filtered by `ind_list`, shaped as
        `[num_cfgs * len(ind_list), num_tau]`.
    """
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
    Converts correlators in the form [Nt, nconf, Ntsrc] to a gvar dataset.

    Args:
        dict_orig_corrs: dict of correlators in the original format
            keys: names - list of strings, giving the key names to corr in corrs
            values: corrs - list of numpy arrays, each of form (Nt, nconf, Ntsrc)
        averages_tsrc (bool): averages over the time sources (3d arrays of corrrelators)
            or not (2d arrays of correlators)

    Returns:
        Correlated gvar dataset with correlators referenced by names
    """
    corr_dict = {}
    for name, corr in dict_orig_corrs.items():
        if averages_tsrc:
            # corr_dict[name] = average_tsrc(corr)
            corr_dict[name] = np.average(corr, axis=-1).transpose() # (n_cf, n_tau)
        else:
            corr_dict[name] = corr.transpose() # (n_cf, n_tau)
    return gv.dataset.avg_data(corr_dict)


def tensor_to_avg_over_tsrc(
    tensor: torch.Tensor, 
    n_tau: int, 
    n_configs: int
) -> npt.NDArray:
    """
    Converts a torch.Tensor into a numpy array and averages over the source times axis.
    
    Args:
        tensor: A PyTorch tensor of shape [num_cfgs * len(ind_list), num_tau]
        
    Returns:
        A 1d numpy array of shape [num_tau] that has been averaged over source times
    """
    temp = tensor.T.reshape((n_tau, n_configs, -1)).numpy()
    return np.average(temp, axis=-1)


# =============================================================================
# LATTICE ROTATING
# =============================================================================
def rotate_sourcetimes(
    corrs: Union[torch.Tensor, npt.NDArray],
    shift: int,
    num_tsrc: Optional[int] = 24
) -> Union[torch.Tensor, npt.NDArray]:
    """
    Rotates the lattice of correlators to reduce autocorrelations across the 
    source times.

    Chooses a value `shift` that is ideally relatively prime to the total
    number of source times. 

    Note: Assumes that the data is shaped `[num_cfgs, num_tsrc, num_tau]`.

    Args:
        corrs: Correlator data
        shift: Shift amount along source time axis
        num_tsrc: Total number of source times in dataset

    Returns:
        A shuffled version of `corrs` where now the source times along each... 
    """
    assert num_tsrc % shift != 0, \
        'Shift should be relatively prime to num_tsrc'

    corrs = corrs.permute(0, 2, 1)  # [configs, taus, tsrc]
    
    # Rotate across the lattice periodically
    for i in range(1, num_tsrc):
        corrs[:, :, i] = corrs[:, :, (i + shift) % num_tsrc ]
    return 
    
def _test_rotate_timesources():
    raise NotImplementedError()


if __name__ == '__main__':  _test_rotate_timesources()

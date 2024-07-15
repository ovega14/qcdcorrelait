import h5py
import numpy as np
import torch

import numpy.typing as npt
from typing import List, Tuple, Optional



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


#===================================================================================================
# DATA NORMALIZATION
#===================================================================================================
def normalize_3d(corr: npt.NDArray) -> Tuple[npt.Array, npt.NDArray]:
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


def normalize_2d(corr: npt.NDArray) -> Tuple[npt.Array, npt.NDArray]:
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


def normalize_1d(corr: npt.Array) -> Tuple[npt.Array, npt.NDArray]:
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
        [num_cfgs * len(ind_list), num_tau].
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

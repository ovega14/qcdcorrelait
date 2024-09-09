"""Input/output utilities for handling files and preparing correlator data."""
import h5py
import numpy as np

import numpy.typing as npt
from typing import Tuple, TypeVar
GVDataset = TypeVar('GVDataset')

from .conversion import tensor_data_by_ind_list
from .normalization import (
    tensor_means_stds_by_axis0,
    normalize_3d
)


# =============================================================================
#  DATA READING AND PREPROCESSING
# =============================================================================
def get_corrs(
    h5fname: str, 
    corr_tags: list[str],
    num_tsrc: int
) -> list[npt.NDArray]:
    """
    Retrieve correlator data from hdf5 cache. Order as (t, conf, tsrc).

    Args:
        h5fname: str, name/location of the hdf5 cache.
        corr_tags: correlator names
        num_tsrc: Number of source times in dataset

    Returns:
        List of reshaped correlators corresponding to `corr_tags`.
    """
    data = h5py.File(h5fname)['data']
    
    result = []
    for tag in corr_tags:
        corr_flat = data[tag]
        np_container = np.zeros(corr_flat.shape)
        corr_flat.read_direct(np_container)
        corr = np_container.transpose()
        corr = corr.reshape((corr_flat.shape[-1], -1, num_tsrc))
        result.append(corr)
    return result


def preprocess_data(
    corr_i: npt.NDArray,
    corr_o: npt.NDArray,
    train_ind_list: list[int],
    bc_ind_list: list[int],
    unlab_ind_list: list[int]
) -> dict[str, npt.NDArray]:
    """
    Preprocesses correlator dataset into a dictionary of input/output data, 
    means, and standard deviations.

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

    corr_i_train_tensor = tensor_data_by_ind_list(corr_i, train_ind_list)
    corr_o_train_tensor = tensor_data_by_ind_list(corr_o, train_ind_list)

    corr_i_bc_tensor = tensor_data_by_ind_list(corr_i, bc_ind_list)
    corr_o_bc_tensor = tensor_data_by_ind_list(corr_o, bc_ind_list)

    corr_i_unlab_tensor = tensor_data_by_ind_list(corr_i, unlab_ind_list)
    corr_o_unlab_tensor = tensor_data_by_ind_list(corr_o, unlab_ind_list)

    # Compute means, stds of training data for normalization & denormalization
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


# =============================================================================
#  MODEL INPUT PREPARATION
# =============================================================================
def prepare_input(
    corr_s: npt.NDArray,
    corr_l: npt.NDArray,
    train_ind_list: list[int],
    bc_ind_list: list[int],
    unlab_ind_list: list[int],
    t_extent: int
) -> Tuple[npt.NDArray, ...]:
    """
    Prepares input data from normalized strange and light correlators for 
    supervised learning.

    The major and secondary axes for "axis-0 of X" are configurations and 
    source times, respectively. Similarly for "axis-0 of Y."

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
    train_ind_list: list[int],
    bc_ind_list: list[int],
    unlab_ind_list: list[int]
) -> Tuple[npt.NDArray, ...]:
    """
    Prepares input data from normalized strange and light correlators for 
    sequence-to-sequence learning.

    The major and secondary axes for "axis-0 of X" are configurations and 
    source times, respectively. Similarly for "axis-0 of Y."

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


def rotate_sourcetimes(
    corrs: npt.NDArray,
    shift: int
) -> npt.NDArray:
    """
    Rotates the correlators across the source times for each time extent to 
    reduce autocorrelations.

    Choose a value `shift` which is relatively prime to the total lattice
    temporal extent, and for each configuration, shift the source times by this
    number of lattice sites.

    Note: Assumes data is shaped as `[num_tau, num_cfg, num_src]`.
    """
    _, num_cfgs, num_src = corrs.shape
    #assert num_src // shift != 0, \
    #    f'Should use shift relatively prime to num_src = {num_src}'

    new_corrs = np.copy(corrs)
    for i in range(1, num_cfgs):
        new_corrs[:, i, :] = np.roll(new_corrs[:, i, :], shift=i*shift, axis=1)
    return new_corrs

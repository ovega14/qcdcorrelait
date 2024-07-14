import h5py
import numpy as np

import numpy.typing as npt
from typing import List, Tuple


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

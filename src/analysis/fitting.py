import corrfitter as cf
import gvar as gv
import numpy as np

import numpy.typing as npt
from typing import Any, Optional, List, TypeVar
Fitter = TypeVar('Fitter')
GVDataset = TypeVar('GVDataset')

import sys
sys.path.insert(0, '../')
from processing.conversion import convert_to_gvars


# =============================================================================
#  ARGUMENTS UTILITIES
# =============================================================================
def corr2_opts(
    tag: str, 
    filename: str, 
    tp: int,
    tmin: int, 
    tmax: int
) -> dict[str, Any]:
    """
    Constructs the dictionary of keyword arguments to be passed to 
    `corrfitter.Corr2()`.
    
    Args:
        tag: 
        filename:
        tp: 
        tmin: Minimum value for time extents used in fitting correlator
        tmax: Maximum value for time extents used in fitting correlator

    Returns:
        Dictionary of fitter arguments
    """
    tdata = range(tmax)
    tfit = range(tmin, tmax)
    opts = {
        'datatag': tag,
        'tp': tp,
        'tdata': tdata,
        'tfit': tfit,
        'a': (filename + ':a', filename + ':ao'),
        'b': (filename + ':a', filename + ':ao'),
        'dE': (filename + ':dE', filename + ':dEo'),
        's': (1., -1.)
    }
    return opts


# =============================================================================
#  CORRELATOR FITTING
# =============================================================================
def fit_corrs(
    dict_corrs: dict[str, npt.NDArray],
    dict_opts: dict[str, Any],
    prior: dict[str, gv.GVar],
    gv_ds: Optional[GVDataset] = None,
    excluding_tags: Optional[List[str]] = []
) -> dict[str, Fitter]:
    """
    Uses `corrfitter` and `lsqfit` to fit the correlation functions to a 
    spectral decomposition in terms of energy splitting and amplitudes.

    Args:
        dict_corrs: Dictionary of correlator data
        dict_opts: Dictionary of keyword arguments for `corrfitter.Corr2()`
        excluding_tags: List of tags to be excluded when performing fits

    Returns:
        Dictionary of fits to the data.
    """
    filename = dict_opts.get('filename')
    tp = dict_opts.get('tp')
    tmin = dict_opts.get('tmin', 10)
    tmax = dict_opts.get('tmax', tp - 10)
    maxit = dict_opts.get('maxit', 5_000)
    averages_tsrc = dict_opts.get('averages_tsrc', False)

    if gv_ds == None:
        data = convert_to_gvars(dict_corrs, averages_tsrc=averages_tsrc)
    else:
        data = gv_ds

    dict_fits = dict()
    for tag in data.keys():
        if tag not in excluding_tags:
            print('tag:', tag)
            model = cf.Corr2(**corr2_opts(
                tag=tag, filename=filename, tp=tp, tmin=tmin, tmax=tmax)
            )
            fitter = cf.CorrFitter(models=[model])
            #p0 = {'a0':0.05398, 'a1':0.0748, 'a2':0.1496, 'a3':0.359, 'a4':0.700,
            #      'dE0':0.39855, 'dE1':0.1960, 'dE2': 0.312, 'dE3': 0.641, 'dE4':1.063,
            #      'ao0': 0.0068, 'ao1': 0.0186, 'ao2': 0.02507, 'ao3': -2.46802e-07,
            #      'dEo0': 0.447, 'dEo1':0.080, 'dEo2':0.307, 'dEo3':0.30}
            p0 = {
                'P5-P5_RW_RW_d_d_m0.548_m0.01555_p000_P5-P5_RW_RW_d_d_m0.164_m0.01555_p000:a':
                    [0.05398, 0.0748, 0.1496, 0.359,  0.700],
                'P5-P5_RW_RW_d_d_m0.548_m0.01555_p000_P5-P5_RW_RW_d_d_m0.164_m0.01555_p000:dE':
                    [0.39855, 0.1960, 0.312, 0.641, 1.063],
                'P5-P5_RW_RW_d_d_m0.548_m0.01555_p000_P5-P5_RW_RW_d_d_m0.164_m0.01555_p000:ao':
                    [0.0068, 0.0186, 0.02507, -2.46802e-07],
                'P5-P5_RW_RW_d_d_m0.548_m0.01555_p000_P5-P5_RW_RW_d_d_m0.164_m0.01555_p000:dEo': 
                    [0.447, 0.080, 0.307, 0.30]
            }
            fit = fitter.lsqfit(data=data, prior=prior, maxit=maxit, p0=p0)
            dict_fits[tag] = fit
            print(fit)
            print('=' * 130)
    return dict_fits

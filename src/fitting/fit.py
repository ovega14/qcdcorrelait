import corrfitter as cf
import gvar as gv
import numpy as np

import numpy.typing as npt
from typing import Any, Optional, List, TypeVar
Fitter = TypeVar('Fitter')
GVDataset = TypeVar('GVDataset')

from .priors import make_prior
import sys
sys.path.insert(0, '../')
from processing.preprocessing import convert_to_gvars


#===================================================================================================
# ARGUMENTS UTILITIES
#===================================================================================================
def corr2_opts(
    tag: str, 
    filename: str, 
    tp: int,
    tmin: int, 
    tmax: int
) -> dict[str, Any]:
    """
    Constructs the dictionary of keyword arguments to be passed to `corrfitter.Corr2()`.
    
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


#===================================================================================================
# CORRELATOR FITTING
#===================================================================================================
def fit_corrs(
    dict_corrs: dict[str, npt.NDArray],
    dict_opts: dict[str, Any],
    gv_ds: GVDataset = None,
    excluding_tags: Optional[List[str]] = []
) -> dict[str, Fitter]:
    """
    Uses `corrfitter` and `lsqfit` to fit the correlation functions to the familiar spectral
    decomposition in terms of energy splitting and amplitudes.

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
    ne = dict_opts.get('ne', 3)
    no = dict_opts.get('no', 3)
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
            model = cf.Corr2(**corr2_opts(tag=tag, filename=filename, tp=tp, tmin=tmin, tmax=tmax))
            prior = make_prior(filename, ne=ne, no=no)
            fitter = cf.CorrFitter(models=[model])
            fit = fitter.lsqfit(data=data, prior=prior, maxit=maxit)
            dict_fits[tag] = fit
            print(fit)
            print('=' * 130)
    return dict_fits

#!/usr/bin/env python3
import torch
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

import argparse
import copy
import json

import numpy.typing as npt
from typing import Any

import sys
sys.path.insert(0, '../src/')
from utils import set_np_seed, set_plot_preferences, save_plot
from processing.io_utils import get_corrs, rotate_sourcetimes
from analysis.fitting import fit_corrs


# =============================================================================
TAU_1: int = 4
TAU_2: int = 12


NCFG: int = 1028
NTAU: int = 192
NSRC: int = 24

SOURCE_TIME_INDS: list[int] = [6]  # for truth fitting only
SHIFT = 8

TAGS: list[str] = [
    'corr_o_truth', 
    'corr_o_pred_corrected', 
    'corr_o_pred_uncorrected'
]
REG_METHOD: str = 'MLP'

# CorrFit hyperparams
NEVEN: int = 5
NODD: int = 4
TPER: int = NTAU
TMIN: int = 2
TMAX: int = NTAU - 2
MAX_ITERS: int = 5_000
AVG_TSRC: bool = False

if __name__ == '__main__':
    print('FITTING CORRELATORS \n data dimensions:')
    print('\t Number of time extents:', NTAU)
    print('\t Number of source times:', NSRC)
    print('\t Number of configurations:', NCFG)

    set_plot_preferences()


# =============================================================================
#  SETTING PRIORS
# =============================================================================
def make_priors(filename: str, *, ne: int, no: int) -> dict[str, gv.GVar]:
    """
    Constructs priors for a given file of correlator data to be used in fits.

    Note: Generally want to use a higher number of odd states than even states.

    Args:
        filename: The name of the subfile containing correlator data
        ne: Number of 'even' or 'non-oscillating' states to include
        no: Number of 'odd' or 'oscillating' states to include
    
    Returns:
        Dictionary of fit parameters and their priors as Gvar objects.
    """
    # TODO: Make this concrete/systematic and less naive
    # TODO: fix numbers of states for other mass combinations
    prior = gv.BufferDict()

    prior[filename + ':a'] = gv.gvar(ne*['0.0(0.5)'])
    prior[filename +':a'][-1] = gv.gvar('0.5(0.5)')
    prior[filename + ':ao'] = gv.gvar(no*['0.0(0.2)'])

    if filename.endswith('P5-P5_RW_RW_d_d_m0.164_m0.01555_p000') or \
       filename.endswith('P5-P5_RW_RW_d_d_m0.164_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.40(5)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(40)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.50(10)', '0.10(10)', '0.20(20)', '0.30(20)', '0.20(20)'][:no])
        prior[filename + ':a'][0] = gv.gvar('0.0500(50)')
    
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.1827_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.1827_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.40(5)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(40)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.50(10)', '0.10(10)', '0.20(20)', '0.30(20)', '0.30(20)'][:no])
        prior[filename + ':a'][0] = gv.gvar('0.0500(50)')
   
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.365_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.365_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.65(5)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(40)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.75(10)', '0.10(10)', '0.20(20)', '0.30(20)', '0.30(20)'][:no])
        prior[filename + ':a'][0] = gv.gvar('0.0500(50)')

    elif filename.endswith('P5-P5_RW_RW_d_d_m0.548_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.548_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.85(5)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(40)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.95(10)', '0.10(10)', '0.20(20)', '0.30(20)', '0.30(20)'][:no])
        prior[filename + ':a'][0] = gv.gvar('0.0500(50)')
    
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.731_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.731_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['1.05(5)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(40)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['1.15(10)', '0.10(10)', '0.20(20)', '0.30(20)', '0.30(20)'][:no])
        prior[filename + ':a'][0] = gv.gvar('0.0500(50)')
    
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.843_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.843_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['1.15(5)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(40)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['1.25(10)', '0.10(10)', '0.20(20)', '0.30(20)', '0.30(20)'][:no])
        prior[filename + ':a'][0] = gv.gvar('0.0500(50)')
    
    else:
        raise KeyError('Unknown file', filename)

    return prior


# =============================================================================
#  FITTING CORRELATORS
# =============================================================================
def make_opts(filename: str) -> dict[str, Any]:
    """Makes dictionary of fit hyperparameters."""
    global NTAU, NEVEN, NODD, TPER, TMIN, TMAX, MAX_ITERS, AVG_TSRC
    
    dict_opts = dict()
    opt_keys: list[str] = [
        'filename', 
        'tp', 
        'tmin', 
        'tmax', 
        'ne', 
        'no', 
        'maxit', 
        'averages_tsrc'
    ]
    opt_vals: list[Any] = [
        filename,
        TPER, 
        TMIN, 
        TMAX, 
        NEVEN, 
        NODD, 
        MAX_ITERS, 
        AVG_TSRC
    ]

    for key, val in zip(opt_keys, opt_vals):
        dict_opts[key] = val
    return dict_opts


def _get_corrs_from_tags(
    dict_data: dict[str, npt.NDArray],
    tags: list[str]
) -> dict[str, npt.NDArray]:
    """Creates dict of correlator data using desired tags."""
    corrs = dict((tag, dict_data[tag]) for tag in tags if tag in dict_data)
    return corrs


def _shuffle_along_axis(x: npt.NDArray, axis: int):
    if axis != 0:
        np.swapaxes(x, axis1=0, axis2=axis)
        np.random.shuffle(x)
        np.swapaxes(x, axis1=0, axis2=axis)
    else:
        np.random.shuffle(x)
    return x



def _load_truth(
    filename: str,  
    output_name: str,
    source_times: list[int] = None
) -> dict[str, npt.NDArray]:
    """
    Loads truth-level data separate from trained data.
    Meant only for use when fitting solely to truth data.

    The data is averaged over the source times axis.
    """
    global NSRC, SHIFT
    corr_o = get_corrs(
        filename,
        [output_name],
        NSRC
    )[0]
    print('corr_o shape:', corr_o.shape)  # [192, 1028, 24]
    
    # rotate and/or shuffle
    corr_o = rotate_sourcetimes(corr_o, shift=SHIFT)

    if source_times is None:
        source_times = list(range(24))
    print('loading truth data with source times:', source_times)
    corr_o = corr_o[..., source_times]
    print('corr_o shape after indexing:', corr_o.shape)

    # Average over source times axis
    corr_o_truth = np.average(corr_o, axis=-1)

    return {'corr_o_truth': corr_o_truth}


def fit_truth_data(source_times: list[int]):
    global NCFG, NSRC, NTAU, TAGS

    dict_data = _load_truth(args.hdf5_filename, 
                            args.output_dataname,
                            source_times)

    dict_orig_corrs = _get_corrs_from_tags(dict_data, TAGS)

    filename = args.output_dataname
    dict_opts = make_opts(filename)
    prior = make_priors(filename, ne=NEVEN, no=NODD)
    dict_fits = fit_corrs(dict_orig_corrs, dict_opts, prior)
    return dict_fits


def nts_curve(filename, results_dir):
    """
    Creates the noies to signal curves for fit parameters derived from truth-
    level data being fit vs number of source times included in the data.
    """
    global NSRC, SHIFT
    a0_nts = []
    dE0_nts = []
    for n in range(1, NSRC+1):
        source_times = np.random.choice(NSRC, size=n, replace=False)
        fit = fit_truth_data(source_times)['corr_o_truth']
        a = fit.p[filename + ':a']
        dE = fit.p[filename + ':dE']
        nts_a  = a[0].sdev / a[0].mean
        nts_dE  = dE[0].sdev / dE[0].mean
        a0_nts.append(nts_a)
        dE0_nts.append(nts_dE)

    fig = plt.figure(figsize=(8., 8.))
    plt.plot(list(range(1, 25)), a0_nts)
    plt.xlabel(r'$N_{\rm src}$')
    plt.ylabel(r'Noise to Signal on $a_0$')
    plt.title(f'shift = {SHIFT}')
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='a0_nts')
    
    fig = plt.figure(figsize=(8., 8.))
    plt.plot(list(range(1, 25)), dE0_nts)
    plt.xlabel(r'$N_{\rm src}$')
    plt.ylabel(r'Noise to Signal on $dE_0$')
    plt.title(f'shift = {SHIFT}')
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='dE0_nts')


def ml_pred_nts():
    global REG_METHOD
    reg_method = REG_METHOD




# =============================================================================
def main(args):
    set_np_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    nts_curve(filename=args.output_dataname, results_dir=args.results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--seed', type=int, default=42)
    add('--hdf5_filename', type=str, 
        default='../data/l64192a_run2_810-6996_1028cfgs.hdf5')
    add('--output_dataname', type=str)
    add('--results_dir', type=str)

    args = parser.parse_args()
    
    with open(args.results_dir + '/data/commandline_args.dat', 'w') as f:
        args_dict = copy.deepcopy(args.__dict__)
        json.dump(args_dict, f, indent=2)

    main(args)

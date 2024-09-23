#!/usr/bin/env python3
import torch
import numpy as np
import gvar as gv

import argparse
import copy
import json

import numpy.typing as npt
from typing import Any

import sys
sys.path.insert(0, '../src/')
from utils import set_np_seed, load_data, set_plot_preferences, save_data
from processing.io_utils import get_corrs, rotate_sourcetimes
from analysis.fitting import fit_corrs
from analysis.tabulation import FitParamsTable


# =============================================================================
NCFG: int = 1028
NTAU: int = 192
NSRC: int = 24

SOURCE_TIME_INDS: list[int] = [6]  # for truth fitting only

TAGS: list[str] = [
    'corr_o_truth', 
    'corr_o_pred_corrected', 
    'corr_o_pred_uncorrected'
]

# CorrFit hyperparams
NEVEN: int = 5
NODD: int = 5
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
    prior[filename + ':ao'] = gv.gvar(no*['0.0(0.2)'])

    if filename.endswith('P5-P5_RW_RW_d_d_m0.164_m0.01555_p000') or \
       filename.endswith('P5-P5_RW_RW_d_d_m0.164_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.400(50)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(20)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.50(10)', '0.10(10)', '0.10(10)', '0.10(10)', '0.20(10)'][:no])
        prior[filename + ':a'][0] = gv.gvar('0.0500(50)')
        if ne >= 5:
            prior[filename +':a'][4] = gv.gvar('0.50(50)')
    
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.1827_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.1827_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.400(50)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(20)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.50(10)', '0.10(10)', '0.10(10)', '0.10(10)', '0.20(10)'][:no])
   
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.365_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.365_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.65(25)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(20)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.75(25)', '0.10(10)', '0.10(10)', '0.10(10)', '0.20(10)'][:no])

    elif filename.endswith('P5-P5_RW_RW_d_d_m0.548_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.548_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.85(05)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(40)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.95(10)', '0.10(10)', '0.20(20)', '0.20(20)', '0.20(20)'][:no])

    elif filename.endswith('P5-P5_RW_RW_d_d_m0.731_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.731_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['1.05(2)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(20)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['1.1(2)', '0.10(10)', '0.10(10)', '0.20(20)', '0.20(20)'][:no])

    elif filename.endswith('P5-P5_RW_RW_d_d_m0.843_m0.01555_p000') or \
         filename.endswith('P5-P5_RW_RW_d_d_m0.843_m0.00311_p000'):
        prior[filename + ':dE'] = gv.gvar(['1.15(2)', '0.200(50)', '0.280(50)', '0.60(20)', '1.00(20)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['1.2(2)', '0.10(10)', '0.10(10)', '0.20(20)', '0.20(20)'][:no])

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
    global NSRC
    corr_o = get_corrs(
        filename,
        [output_name],
        NSRC
    )[0]
    print('corr_o shape:', corr_o.shape)  # [192, 1028, 24]
    corr_o = rotate_sourcetimes(corr_o, shift=7)
    if source_times is None:
        source_times = list(range(24))
    print('loading truth data with source times:', source_times)
    corr_o = corr_o[..., source_times]
    print('corr_o shape after indexing:', corr_o.shape)

    # Average over source times axis
    corr_o_truth = np.average(corr_o, axis=-1)

    return {'corr_o_truth': corr_o_truth}


# =============================================================================
def main(args):
    set_np_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    global NCFG, NSRC, NTAU, TAGS

    try:
        dict_data = load_data(args.results_dir + '/dict_data')
    except FileNotFoundError:
        global SOURCE_TIME_INDS
        dict_data = _load_truth(args.hdf5_filename, 
                                args.output_dataname,
                                SOURCE_TIME_INDS)

    if args.compare_ratio_method:
        ds_ratio_method = load_data(args.results_dir + '/ds_ratio_method')
    if args.compare_ml_ratio_method:
        ds_ml_ratio_method = load_data(args.results_dir + '/ds_ml_ratio_method')

    dict_orig_corrs = _get_corrs_from_tags(dict_data, TAGS)

    if args.input_dataname:
        filename = args.input_dataname + '_' + args.output_dataname
    else:
        filename = args.output_dataname
    dict_opts = make_opts(filename)
    prior = make_priors(filename, ne=NEVEN, no=NODD)
    dict_fits = fit_corrs(dict_orig_corrs, dict_opts, prior)
    all_dict_fits = dict_fits.copy()

    # Write Results -----------------------------------------------------------
    with open(args.results_dir + '/results/fits.txt', 'w') as f:
        for tag in dict_fits.keys():
            print(tag, file=f)
            print(dict_fits[tag], file=f)
    with open(args.results_dir + '/results/latex_table.txt', 'w') as f:
        for tag in dict_fits.keys():
            print(tag + ':\n', file=f)
            print(FitParamsTable.write_line(
                args.reg_method, dict_fits, filename, tag), file=f
            )
            print('=' * 120, file=f)

    if args.compare_ratio_method == 1:
        dict_fits = fit_corrs(
            dict_corrs = None,
            dict_opts = dict_opts,
            prior = prior,
            gv_ds = ds_ratio_method,
            excluding_tags=['hp_i', 'lp_i']
        )
        with open(args.results_dir + '/results/fits.txt', 'a') as f:
            for tag in dict_fits.keys():
                print(tag, file=f)
                print(dict_fits[tag], file=f)
        with open(args.results_dir + '/results/latex_table.txt', 'a') as f:
            for tag in dict_fits.keys():
                if tag == 'ratio_method_pred':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(
                        'RM', dict_fits, filename, tag), file=f
                    )
                elif tag == 'ratio_method_pred_modified':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(
                        'bRM', dict_fits, filename, tag), file=f
                    )
                else:
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(
                        args.reg_method, dict_fits, filename, tag), file=f
                    )
                print('=' * 120, file=f)
        all_dict_fits.update(dict_fits)
    
    if args.compare_ml_ratio_method == 1:
        dict_fits = fit_corrs(
            dict_corrs = None,
            dict_opts = dict_opts,
            prior = prior,
            gv_ds = ds_ml_ratio_method,
            excluding_tags=['hp_o', 'hp_i', 'lp_i', 'lp_o']
        )
        with open(args.results_dir + '/results/fits.txt', 'a') as f:
            for tag in dict_fits.keys():
                print(tag, file=f)
                print(dict_fits[tag], file=f)
        with open(args.results_dir + '/results/latex_table.txt', 'a') as f:
            for tag in dict_fits.keys():
                if tag == 'ml_ratio_method_pred':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(
                        'RM + ML', dict_fits, filename, tag), file=f
                    )
                elif tag == 'ml_ratio_method_pred_modified':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(
                        'bRM + ML', dict_fits, filename, tag), file=f
                    )
                else:
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(
                        args.reg_method, dict_fits, filename, tag), file=f
                    )
                print('=' * 120, file=f)
        all_dict_fits.update(dict_fits)

    save_data(all_dict_fits, args.results_dir + '/all_dict_fits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--seed', type=int, default=42)
    add('--reg_method', type=str, default='MLP')  # TODO: remove this arg
    add('--compare_ratio_method', type=int, default=0)
    add('--compare_ml_ratio_method', type=int, default=0)
    add('--hdf5_filename', type=str, 
        default='../data/l64192a_run2_810-6996_1028cfgs.hdf5')
    add('--input_dataname', type=str)
    add('--output_dataname', type=str)
    add('--results_dir', type=str)

    args = parser.parse_args()
    
    with open(args.results_dir + '/data/commandline_args.dat', 'w') as f:
        args_dict = copy.deepcopy(args.__dict__)
        json.dump(args_dict, f, indent=2)

    main(args)

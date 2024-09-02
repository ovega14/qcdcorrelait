#!/usr/bin/env python3
import torch
import gvar as gv

import argparse
import copy
import json

import numpy.typing as npt
from typing import Any

import sys
sys.path.insert(0, '../src/')
from utils import set_np_seed, save_data, load_data
from analysis.fitting import fit_corrs
from analysis.tabulation import FitParamsTable


# =============================================================================
NCFG: int = 1028
NTAU: int = 192
NSRC: int = 24

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
    prior[filename + ':ao'] = gv.gvar(no*['0.0(0.5)'])

    if filename.endswith('P5-P5_RW_RW_d_d_m0.164_m0.01555_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.400(5)', '0.20(5)', '0.28(5)', '0.6(2)', '1.0(2)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.25(5)', '0.25(5)', '0.10(10)', '0.10(10)', '0.20(10)'][:no])
        prior[filename + ':a'][0] = gv.gvar('0.050(5)')
        if ne >= 5:
            prior[filename +':a'][4] = gv.gvar('0.5(5)')
    
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.1827_m0.01555_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.40(5)', '0.20(5)', '0.20(10)'])
        prior[filename + ':dEo'] = gv.gvar(['0.30(5)', '0.20(5)', '0.20(10)'])
    
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.365_m0.01555_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.65(25)', '0.27(5)', '0.62(5)'])
        prior[filename + ':dEo'] = gv.gvar(['0.65(25)', '0.06(10)', '0.28(5)'])

    elif filename.endswith('P5-P5_RW_RW_d_d_m0.548_m0.01555_p000'):
        prior[filename + ':dEo'] = gv.gvar(['0.85(05)', '0.10(5)', '0.25(10)', '0.25(30)', '0.25(30)'])
        prior[filename + ':dE'] = gv.gvar(['0.85(05)', '0.20(5)', '0.30(10)', '0.30(20)', '0.30(20)'])

    elif filename.endswith('P5-P5_RW_RW_d_d_m0.843_m0.01555_p000'):
        prior[filename + ':dE'] = gv.gvar(['1.15(2)', '0.11(5)', '0.25(10)'])
        prior[filename + ':dEo'] = gv.gvar(['1.2(2)', '0.160(5)', '0.19(10)'])

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


# =============================================================================
def main(args):
    set_np_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    global NCFG, NSRC, NTAU, TAGS

    dict_data = load_data(args.results_dir + '/dict_data')
    ds_ratio_method = load_data(args.results_dir + '/ds_ratio_method')
    ds_ml_ratio_method = load_data(args.results_dir + '/ds_ml_ratio_method')

    dict_orig_corrs = _get_corrs_from_tags(dict_data, TAGS)

    filename = args.input_dataname + '_' + args.output_dataname
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
                if tag == 'ratio_method_pred':
                    print("ml_ratio_method_pred", file=f)
                elif tag == 'ratio_method_pred_modified':
                    print("ml_ratio_method_pred_modified", file=f)
                else:
                    print(tag, file=f)
                print(dict_fits[tag], file=f)
        with open(args.results_dir + '/results/latex_table.txt', 'a') as f:
            for tag in dict_fits.keys():
                if tag == 'ratio_method_pred':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(
                        'RM + ML', dict_fits, filename, tag), file=f
                    )
                elif tag == 'ratio_method_pred_modified':
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

    #save_data(all_dict_fits, args.results_dir + '/all_dict_fits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--seed', type=int, default=42)
    add('--reg_method', type=str, default='MLP')  # TODO: remove this arg
    add('--compare_ratio_method', type=bool, default=1)
    add('--compare_ml_ratio_method', type=bool, default=1)
    add('--input_dataname', type=str)
    add('--output_dataname', type=str)
    add('--results_dir', type=str)

    args = parser.parse_args()
    
    with open(args.results_dir + '/data/commandline_args.dat', 'w') as f:
        args_dict = copy.deepcopy(args.__dict__)
        json.dump(args_dict, f, indent=2)

    main(args)

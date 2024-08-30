#!/usr/bin/env python3
import torch
import gvar as gv
import argparse

from typing import Any

import sys
sys.path.insert('../src/')
from utils import load_model, set_np_seed, save_data, load_data
from regression.inference import predict


# =============================================================================
NCFG: int = 1028
NTAU: int = 192
NSRC: int = 24

# CorrFit hyperparams
NEVEN: int = 5
NODD: int = 5
TPER: int = NTAU
TMIN: int = 2
TMAX: int = NTAU - TPER
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

filename = dict_opts.get('filename')
dict_fits = fit_corrs(dict_orig_corrs, dict_opts)
all_dict_fits = dict_fits.copy()


# =============================================================================
def main(args):
    seed = args.seed
    set_np_seed(seed)
    torch.set_default_dtype(torch.float64)

    global NCFG, NSRC, NTAU

    model = load_model(args.results_dir + '/model')
    dict_data = load_data(args.results_dir + '/dict_data')

    num = dict_data["corr_i_train_tensor"] - dict_data["corr_i_train_means"]
    denom = dict_data["corr_i_train_stds"]
    n_corr_i_train_tensor = num / denom
    
    dict_results = predict(
        n_corr_i_train_tensor = n_corr_i_train_tensor,
        model = model,
        reg_method = args.reg_method,
        dict_data = dict_data
    )
    if args.save_results == 1:
        save_data(dict_results, path=args.results_dir + '/results')

    dict_data["n_corr_o_train_tensor"] = (dict_data["corr_o_train_tensor"] - dict_data["corr_o_train_means"]) / dict_data["corr_o_train_stds"]

    dict_data["n_corr_i_unlab_tensor"] = (dict_data["corr_i_unlab_tensor"] - dict_data["corr_i_train_means"]) / dict_data["corr_i_train_stds"]
    dict_data["n_corr_o_unlab_tensor"] = (dict_data["corr_o_unlab_tensor"] - dict_data["corr_o_train_means"]) / dict_data["corr_o_train_stds"]

    dict_data["n_corr_i_bc_tensor"] = (dict_data["corr_i_bc_tensor"] -  dict_data["corr_i_train_means"]) /  dict_data["corr_i_train_stds"]
    dict_data["n_corr_o_bc_tensor"] = (dict_data["corr_o_bc_tensor"] -  dict_data["corr_o_train_means"]) /  dict_data["corr_o_train_stds"]

    fits = analysis_pred(
        corr_i, corr_o,
        NTAU, NCFG, NSRC,
        dict_data,
        dict_results,
        args.reg_method,
        args.results_dir,
        args
    )
    return fits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--seed', type=int, default=42)
    add('--reg_method', type=str, default='MLP')  # TODO: remove this arg
    add('--save_results', type=bool, default=0)
    add('--results_dir', type=str)

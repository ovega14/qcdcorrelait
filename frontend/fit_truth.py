#!/usr/bin/env python3
import torch
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

import argparse
import copy
import json

import numpy.typing as npt
from typing import Any, TypeVar

import sys
sys.path.insert(0, '../src/')
from utils import set_np_seed, set_plot_preferences, save_plot
from processing.io_utils import get_corrs, rotate_sourcetimes, preprocess_data
from processing.conversion import tensor_to_avg_over_tsrc
from analysis.fitting import fit_corrs
from regression.torch_regressors import *
from regression.inference import predict
from regression.utils import adjust_learning_rate, l2_regularization


# =============================================================================
TorchRegressor = TypeVar('TorchRegressor')
SklearnRegressor = TypeVar('SklearnRegressor')

TORCH_REGRESSORS: dict[str, TorchRegressor] = {
    'Linear': LinearModel,
    'MLP': MLP,
    'CNN': CNN,
    'Transformer': Transformer,
    'Identity': torch.nn.Identity
}

TAU_1: int = 4
TAU_2: int = 12


NCFG: int = 1028
NTAU: int = 192
NSRC: int = 24

# ###
# For truth fitting only
SOURCE_TIME_INDS: list[int] = list(range(24)) 
SHIFT: int = 3
# ###

###
# For ML fitting only
NTRAIN: int = 1
NBC: int = 5
###

TAGS: list[str] = [
    'corr_o_truth', 
    'corr_o_pred_corrected', 
    'corr_o_pred_uncorrected'
]

# QUARK MASSES
STRANGE_MASS: float = 0.01555
LIGHT_MASS: float = 0.00311
HEAVY_MASSES: list[float] = [
    0.164,
    0.1827,
    0.365,
    0.548,
    0.731,
    0.843
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
    print('RUNNING METRICS \n data dimensions:')
    print('\t Number of time extents:', NTAU)
    print('\t Number of source times:', NSRC)
    print('\t Number of configurations:', NCFG)
    print('='*120)

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
    source_times: list[int],
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
    #print('loading truth data with source times:', source_times)
    corr_o = corr_o[..., source_times]
    print('corr_o shape after indexing:', corr_o.shape)

    # Average over source times axis
    corr_o_truth = np.average(corr_o, axis=-1)
    return {'corr_o_truth': corr_o_truth}


def prepare_data(
    dict_data: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardizes the two-point strange (input) and light (output) correlator 
    data for input to a model for training.

    Args:
        dict_data: Dictionary of pre-processed correlator data

    Returns:
        Training input and output data
    """
    input_train = dict_data["corr_i_train_tensor"]
    output_train = dict_data["corr_o_train_tensor"]
    input_train_means = dict_data["corr_i_train_means"]
    output_train_means = dict_data["corr_o_train_means"]
    input_train_stds = dict_data["corr_i_train_stds"]
    output_train_stds = dict_data["corr_o_train_stds"]

    corr_2pt_s_train = (input_train - input_train_means) / input_train_stds
    corr_2pt_l_train = (output_train - output_train_means) / output_train_stds

    return corr_2pt_s_train, corr_2pt_l_train


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


def fit_ml_data(dict_data):
    dict_orig_corrs = _get_corrs_from_tags(dict_data, TAGS)

    filename = args.output_dataname
    dict_opts = make_opts(filename)
    prior = make_priors(filename, ne=NEVEN, no=NODD)
    dict_fits = fit_corrs(dict_orig_corrs, dict_opts, prior)
    return dict_fits


# =============================================================================
#  MACHINE LEARNING
# =============================================================================
def make_model(
    reg_method: str,
    seed: int
) -> Union[TorchRegressor, SklearnRegressor, list[SklearnRegressor]]:
    """
    Prepares a regression model to be trained.

    Args:
        reg_method: Name of method to use for regression
        seed: Integer seed to use for reproducibility

    Returns:
        model: An initialized PyTorch or SkLearn regressor
    """
    torch.manual_seed(seed)
    global NTAU

    if reg_method in TORCH_REGRESSORS.keys():
        print(f'Using {reg_method} for regression.')
        if reg_method == 'MLP':
            model = MLP(
                input_dim = NTAU, 
                output_dim = NTAU, 
                hidden_dims = [NTAU // 4], 
                batch_norm = False
            )
        elif reg_method == 'Linear':
            model = LinearModel(
                input_dim = NTAU, 
                output_dim = NTAU
            )
        elif reg_method == 'CNN':
            model = CNN(
                in_channels = 1, 
                out_channels = 1, 
                hidden_channels = [1], 
                kernel_size = 15, 
                batch_norm = False
            )
        elif reg_method == 'Transformer':
            model = Transformer(
                input_dim = 1, 
                num_heads = 1
            )
        elif reg_method == 'Identity':
            model = torch.nn.Identity
    else:
        raise KeyError(f"Unknown regression method '{reg_method}'.")
    return model


def loss_func(
    prediction: torch.Tensor, 
    target: torch.Tensor,
    l2_coeff: float,
    model: torch.nn.Module
) -> torch.FloatTensor:
    """
    MSE loss with :math:`\ell_2` regularization. 

    Args:
        prediction: Data predicted by model
        target: Desired output data
        l2_coeff: Coefficient for regularization
        model: Neural net being trained

    Returns:
        Scalar-valued loss
    """
    loss = F.mse_loss(prediction, target)
    l2_reg = l2_regularization(l2_coeff, model)
    
    return loss + l2_reg


def train_torch_network(
    input_data: torch.Tensor, 
    output_data: torch.Tensor,
    lr: float,
    l2_coeff: float,
    training_steps: int,
    model: TorchRegressor,
    results_dir: str
) -> TorchRegressor:
    """
    Trains the neural network.
    
    For neural networks implemented via PyTorch, training is done according to 
    MSE loss with :math:`\ell^2` regularization. The training loss curve is 
    saved and plotted.

    Args:
        input_data: Input training data
        output_data: Output training data
        lr: Base learning rate
        l2_coeff: Regularization coefficient
        training_steps: Number of iterations for training
        model: Neural net to be trained
        results_dir: Directory in which plots from training will be saved
    
    Returns:
        Trained model
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for i in range(training_steps):
        lr2 = adjust_learning_rate(training_steps, 0.3, lr, optimizer, i)

        prediction = model(input_data)
        loss = loss_func(prediction, output_data, l2_coeff, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Step:', i)
            print(f'Loss: {loss.item():.12f} | lr: {lr2:.12f}')
        losses.append(loss.item())
    
    return model


def infer_ml_data(
    input_dataname: str, 
    output_dataname: str,
    train_ind_list: list[int],
    bc_ind_list: list[int],
    unlab_ind_list: list[int],
    reg_method: str
) -> dict[str, npt.NDArray]:
    """Fits the ML model and then fits inferred data"""
    global NSRC, NTRAIN, NBC

    print('infer ml data using reg_method', reg_method)
    corr_i, corr_o = get_corrs(
        args.hdf5_filename,
        [input_dataname, output_dataname],
        NSRC
    )

    corr_i = rotate_sourcetimes(corr_i, shift=SHIFT)
    corr_o = rotate_sourcetimes(corr_o, shift=SHIFT)
          
    dict_data = preprocess_data(
        corr_i, corr_o,
        train_ind_list,
        bc_ind_list,
        unlab_ind_list
    )

    inputs, outputs = prepare_data(dict_data)
    
    model = make_model(reg_method, args.seed)

    if reg_method == 'Identity':
        trained_model = model
    else:
        trained_model = train_torch_network(
            inputs, outputs,
            lr = args.lr,
            l2_coeff = args.l2_coeff,
            training_steps = args.training_steps,
            model = model,
            results_dir = args.results_dir
        )

    n_corr_i_train_tensor = (dict_data["corr_i_train_tensor"] - dict_data["corr_i_train_means"]) / dict_data["corr_i_train_stds"]
    
    dict_results = predict(
        n_corr_i_train_tensor = n_corr_i_train_tensor,
        model = trained_model,
        reg_method = reg_method,
        dict_data = dict_data
    )

    dict_data["n_corr_o_train_tensor"] = (dict_data["corr_o_train_tensor"] - dict_data["corr_o_train_means"]) / dict_data["corr_o_train_stds"]

    dict_data["n_corr_i_unlab_tensor"] = (dict_data["corr_i_unlab_tensor"] - dict_data["corr_i_train_means"]) / dict_data["corr_i_train_stds"]
    dict_data["n_corr_o_unlab_tensor"] = (dict_data["corr_o_unlab_tensor"] - dict_data["corr_o_train_means"]) / dict_data["corr_o_train_stds"]

    dict_data["n_corr_i_bc_tensor"] = (dict_data["corr_i_bc_tensor"] -  dict_data["corr_i_train_means"]) /  dict_data["corr_i_train_stds"]
    dict_data["n_corr_o_bc_tensor"] = (dict_data["corr_o_bc_tensor"] -  dict_data["corr_o_train_means"]) /  dict_data["corr_o_train_stds"]

    n_corr_o_unlab_tensor = dict_data["n_corr_o_unlab_tensor"]
    n_corr_o_bc_tensor = dict_data["n_corr_o_bc_tensor"]
    corr_o_train_stds = dict_data["corr_o_train_stds"]
    corr_o_train_means = dict_data["corr_o_train_means"]

    corr_o_train_pred_tensor = dict_results["corr_o_train_pred_tensor"]
    corr_o_unlab_pred_tensor = dict_results["corr_o_unlab_pred_tensor"]
    corr_o_bc_pred_tensor = dict_results["corr_o_bc_pred_tensor"]
    n_corr_o_unlab_pred_tensor = dict_results["n_corr_o_unlab_pred_tensor"]
    n_corr_o_bc_pred_tensor = dict_results["n_corr_o_bc_pred_tensor"]

    corr_o_train_pred = corr_o_train_pred_tensor.T.reshape((NTAU, NCFG, -1)).numpy()
    corr_o_bc_pred = corr_o_bc_pred_tensor.T.reshape((NTAU, NCFG, -1)).numpy()
    corr_o_unlab_pred = corr_o_unlab_pred_tensor.T.reshape((NTAU, NCFG, -1)).numpy()

    corr_o_pred = []
    curr_ind_train = 0
    curr_ind_bc = 0
    curr_ind_unlab = 0
    # Separate predicted data by subsets again
    for i in range(NSRC):
        if i in train_ind_list:
            corr_o_pred.append(corr_o_train_pred[:, :, curr_ind_train])
            curr_ind_train += 1
        elif i in bc_ind_list:
            corr_o_pred.append(corr_o_bc_pred[:, :, curr_ind_bc])
            curr_ind_bc += 1
        else:
            corr_o_pred.append(corr_o_unlab_pred[:, :, curr_ind_unlab])
            curr_ind_unlab += 1
    corr_o_pred = np.array(corr_o_pred)
    corr_o_pred = np.swapaxes(corr_o_pred, 0, 1)
    corr_o_pred = np.swapaxes(corr_o_pred, 1, 2)
    
    n_corr_o_unlab_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_unlab_tensor, NTAU, NCFG)
    n_corr_o_unlab_pred_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_unlab_pred_tensor, NTAU, NCFG)

    n_corr_o_bc_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_bc_tensor, NTAU, NCFG)
    n_corr_o_bc_pred_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_bc_pred_tensor, NTAU, NCFG)

    n_corr_o_uncorrected = n_corr_o_unlab_pred_vs_tau
    n_corr_o_corrected = n_corr_o_uncorrected + n_corr_o_bc_vs_tau - n_corr_o_bc_pred_vs_tau
    n_bias_correction_vs_tau = n_corr_o_bc_vs_tau - n_corr_o_bc_pred_vs_tau

    corr_o_pred_corrected = n_corr_o_corrected * corr_o_train_stds[:, None] + corr_o_train_means[:, None]
    corr_o_pred_uncorrected = n_corr_o_uncorrected * corr_o_train_stds[:, np.newaxis] + corr_o_train_means[:, np.newaxis]
    corr_o_bias_correction = n_bias_correction_vs_tau * corr_o_train_stds[:, np.newaxis] + corr_o_train_means[:, np.newaxis]
    
    corr_o_truth = np.average(corr_o, axis=-1)
    corr_o_train_truth = np.average(corr_o[:, :, train_ind_list], axis=-1)
    corr_o_labeled_truth = np.average(corr_o[:, :, train_ind_list + bc_ind_list], axis=-1)

    dict_data['corr_o_truth'] = corr_o_truth
    dict_data['corr_o_pred_corrected'] = corr_o_pred_corrected
    dict_data['corr_o_pred_uncorrected'] = corr_o_pred_uncorrected

    return dict_data


# =============================================================================
#  NOISE TO SIGNAL / METRICS
# =============================================================================
def correlator_nts(source_times: list[int], plot: bool = False) -> npt.NDArray:
    """
    For some nominal values `TAU_1` and `TAU_2` (globally defined), looks at
    the noies to signal of the correlator at these time extent values
    """
    global TAU_1, TAU_2

    truth_data = _load_truth(
        args.hdf5_filename, 
        args.output_dataname,
        source_times=source_times
    )['corr_o_truth']    
    print('truth_data shape:', truth_data.shape)  # [NTAU, NCFG]
    truth_data = truth_data.T  # [NCFG, NTAU]
    nts = np.std(truth_data, axis=0) / np.average(truth_data, axis=0)
    print(f'Noise to signal at {TAU_1}:', nts[TAU_1])
    print(f'Noise to signal at {TAU_2}:', nts[TAU_2])
    if plot:
        fig = plt.figure(figsize=(8., 8.))
        plt.plot(nts)
        plt.xlabel(r'Time Extent, $\tau$')
        plt.ylabel(r'Noise to Signal Ratio of $C(\tau)$')
        save_plot(fig=fig, path=f'{args.results_dir}/plots/', filename='corr_nts_vs_tau')
    return nts


def correlator_nts_vs_nsrc():
    """
    For the globally defined nominal values of `TAU_1` and `TAU_2`, creates a
    plot of the noise to signal on the correlator vs number of source times 
    in the truth-level dataset.
    """
    global NSRC, SHIFT, TAU_1, TAU_2

    nts_tau1s = []
    nts_tau2s = []
    for n in range(1, NSRC+1):
        source_times = np.random.choice(NSRC, size=n, replace=False)
        nts = correlator_nts(source_times)
        nts_tau1, nts_tau2 = nts[TAU_1], nts[TAU_2]
        nts_tau1s.append(nts_tau1)
        nts_tau2s.append(nts_tau2)
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(nts_tau1s, label=rf'$\tau =$ {TAU_1}')
    plt.plot(nts_tau2s, label=rf'$\tau =$ {TAU_2}')
    plt.xlabel(r'$N_{\rm src}$')
    plt.ylabel(r'Noise to signal on $C(\tau)$')
    plt.legend()
    save_plot(fig=fig, path=f'{args.results_dir}/plots/', filename='corr_nts_vs_nsrc')


def fit_params_nts(filename: str, dict_data, results_dir):
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

    ml_fit = fit_ml_data(dict_data)['corr_o_pred_corrected']
    a = ml_fit.p[filename + ':a']
    dE = ml_fit.p[filename + ':dE']
    nts_a  = a[0].sdev / a[0].mean
    nts_dE  = dE[0].sdev / dE[0].mean

    fig = plt.figure(figsize=(8., 8.))
    plt.plot(list(range(1, 25)), a0_nts, label='Truth', color='blue')
    plt.hlines(y=nts_a, xmin=1, xmax=25, label='ML', color='red')
    plt.xlabel(r'$N_{\rm src}$')
    plt.ylabel(r'Noise to Signal on $a_0$')
    plt.title(rf'shift = {SHIFT}, N_train = {NTRAIN}, N_BC = {NBC}')
    plt.legend()
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='a0_nts')
    
    fig = plt.figure(figsize=(8., 8.))
    plt.plot(list(range(1, 25)), dE0_nts, label='Truth', color='blue')
    plt.hlines(y=nts_dE, xmin=1, xmax=25, label='ML', color='red')
    plt.xlabel(r'$N_{\rm src}$')
    plt.ylabel(r'Noise to Signal on $dE_0$')
    plt.title(rf'shift = {SHIFT}, N_train = {NTRAIN}, N_BC = {NBC}')
    plt.legend()
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='dE0_nts')


# =============================================================================
#  METRICS
# =============================================================================
def combined_obs(dict_data):
    """
    Weighted average of observable over the BC and LD subsets.
    
    Implemented as in equation (16) of [arxiv 1909.10990].
    """
    obs_pred_bc = dict_data['corr_o_pred_corrected']
    raise NotImplementedError('Ignored for now...')
    


def cost_metric(dict_data):
    """
    Cost function as implemented in equation (15) of [arxiv 1909.10990],
    where the term `N_in` is not included.

    Note: for now, we take the 'sigma_comb' to just be the variance of the
    bias-corrected, ML prediction instead of a weighted average over subsets.
    """
    global NTRAIN, NBC, NSRC, TAU_1

    num_ld = NTRAIN + NBC
    num_uld = NSRC - num_ld

    
    corr_ul = tensor_to_avg_over_tsrc(dict_data['corr_o_unlab_tensor'], NTAU, NCFG)
    corr_pred_bc = dict_data['corr_o_pred_corrected']
    print('corr_ul.shape:', corr_ul.shape)
    print('corr_o_pred_corrected.shape:', corr_pred_bc.shape)
    ratio = np.var(corr_ul, axis=-1) / np.var(corr_pred_bc, axis=-1)
    num_eff = ratio * num_uld
    metric = (NBC + NTRAIN) / num_eff

    print('METRICS at TAU =', TAU_1)
    print('\tEFFECTIVE NUMBER OF SOURCE TIMES:', num_eff[TAU_1])
    print('\tCOST METRIC:', metric[TAU_1])

    # Compare numbers in denominator
    ts = 986
    td = 4960
    N_in = 24

    print('\tt_s * N_in:', ts * N_in)
    print('\ttd * N_ul * ratio', td * num_uld * ratio[TAU_1])
    ts_times_N_in = ts * N_in  # constant
    td_nul_ratio = td * num_uld * ratio
    fig = plt.figure(figsize=(8, 8))
    plt.hlines(y=ts_times_N_in, xmin=0, xmax=NTAU, label=rf'$t_s N_\mathrm{{in}}$', colors='red')
    plt.plot(td_nul_ratio, label=rf'$t_d N_\mathrm{{ul}} \frac{{\sigma^2_\mathrm{{ul}}}}{{\sigma^2_\mathrm{{comb}}}}$')
    plt.xlabel(rf'$\tau$')
    plt.ylabel(rf'$N_{{\rm eff}}$')
    plt.legend(loc='upper center', frameon=False)
    save_plot(fig, path=f'{args.results_dir}/plots/', filename='Nsrc_vs_tau')

    return metric


# =============================================================================
def main(args):
    global STRANGE_MASS, HEAVY_MASSES
    global SOURCE_TIME_INDS
    global NSRC, NTRAIN, NBC

    global REG_METHOD
    reg_method = REG_METHOD

    set_np_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    # Choose source time subsets
    total_inds: set[int] = {n for n in range(24)}
    labeled_inds = np.random.choice(NSRC, size=NTRAIN + NBC, replace=False)
    unlab_inds: set[int] = list(total_inds - set(labeled_inds))
    bc_inds = np.random.choice(labeled_inds, size=NBC, replace=False)
    train_inds = set(labeled_inds) - set(bc_inds)

    train_ind_list = list(train_inds)
    bc_ind_list = list(bc_inds)
    unlab_ind_list = list(unlab_inds)
    print('SOURCE TIME ALLOCATIONS:')
    print(f'NTRAIN = {NTRAIN}, NBC = {NBC}, NUL = {NSRC - NTRAIN - NBC}')
    print('Training indices:', train_ind_list)
    print('Bias correction indices:', bc_ind_list)
    print('Testing indices:', unlab_ind_list)

    # Compare cost metric for different mass combinations
    def compare_heavy_masses() -> None:
        mi2 = STRANGE_MASS
        mo2 = STRANGE_MASS
        mo1 = 0.164
        
        colors = plt.cm.jet(np.linspace(0, 1, len(HEAVY_MASSES)))
        fig = plt.figure(figsize=(8, 8))
        print('-'*120)
        print('HEAVY MASS COMPARISON')
        for i, mi1 in enumerate(HEAVY_MASSES):
            print()
            print(f'Input Masses: mi1 = {mi1}, mi2 = {mi2}')
            print(f'Output Masses: mo1 = {mo1}, mo2 = {mo2}')

            input_dataname = f'P5-P5_RW_RW_d_d_m{mi1}_m{mi2}_p000'
            output_dataname = f'P5-P5_RW_RW_d_d_m{mo1}_m{mo2}_p000'

            dict_data = infer_ml_data(input_dataname, output_dataname, 
                train_ind_list, bc_ind_list, unlab_ind_list,
                reg_method)
            metric = cost_metric(dict_data)
            plt.plot(metric, lw=0.75, color=colors[i], label=rf'{mi1}')
        plt.hlines(1.0, xmin=0, xmax=NTAU, color='black', ls='dashed', lw=0.75)
        plt.xlabel(r'Time Extent, $\tau$')
        plt.ylabel('Cost')
        plt.text(x=0.1, y=1.4, 
                 s=rf'$(m_h, {mi2}) \rightarrow ({mo1}, {mo2})$',
                 fontsize='small')
        plt.title(f'Heavy to Heavy Prediction Costs with {reg_method}')
        plt.legend(frameon=False, loc='best', prop={'size': 12}, title=r'$a m_h$')
        save_plot(fig, path=f'{args.results_dir}/plots/', filename='cost_metric_vs_tau')

    compare_heavy_masses()

    # Strange to light comparison
    def compare_strange_to_light() -> None:
        mi2 = STRANGE_MASS
        mo2 = LIGHT_MASS
        
        colors = plt.cm.jet(np.linspace(0, 1, len(HEAVY_MASSES)))
        fig = plt.figure(figsize=(8, 8))
        print('-'*120)
        print('STRANGE TO LIGHT COMPARISON')
        for i, (mi1, mo1) in enumerate(zip(HEAVY_MASSES, HEAVY_MASSES)):
            print()
            print(f'Input Masses: mi1 = {mi1}, mi2 = {mi2}')
            print(f'Output Masses: mo1 = {mo1}, mo2 = {mo2}')

            input_dataname = f'P5-P5_RW_RW_d_d_m{mi1}_m{mi2}_p000'
            output_dataname = f'P5-P5_RW_RW_d_d_m{mo1}_m{mo2}_p000'

            dict_data = infer_ml_data(input_dataname, output_dataname, 
                train_ind_list, bc_ind_list, unlab_ind_list,
                reg_method)
            metric = cost_metric(dict_data)
            plt.plot(metric, lw=0.75, color=colors[i], label=f'{mi1}')
        plt.hlines(1.0, xmin=0, xmax=NTAU, color='black', ls='dashed', lw=0.75)
        plt.xlabel(r'Time Extent, $\tau$')
        plt.ylabel('Cost')
        plt.text(x=0.1, y=1.4, 
                 s=rf'$(m_h, {mi2}) \rightarrow (m_h, {mo2})$',
                 fontsize='small')
        plt.title(f'Strange to Light Prediction Costs with {reg_method}')
        plt.legend(frameon=False, loc='best', prop={'size': 12}, title=r'$a m_h$')
        save_plot(fig, path=f'{args.results_dir}/plots/', filename='strange2light_cost_metric')

    compare_strange_to_light()

    # Compare cost of ML vs Identity (baseline)
    def compare_ml_vs_identity() -> None:
        #mi1 = 0.548
        #mi2 = STRANGE_MASS
        #mo1 = 0.164
        #mo2 = STRANGE_MASS
        mi1 = 0.843
        mi2 = STRANGE_MASS
        mo1 = 0.843
        mo2 = LIGHT_MASS
        print(f'Input Masses: mi1 = {mi1}, mi2 = {mi2}')
        print(f'Output Masses: mo1 = {mo1}, mo2 = {mo2}')
        
        input_dataname = f'P5-P5_RW_RW_d_d_m{mi1}_m{mi2}_p000'
        output_dataname = f'P5-P5_RW_RW_d_d_m{mo1}_m{mo2}_p000'

        fig = plt.figure(figsize=(8, 8))
        print('-'*120)
        print('REG METHODS COMPARISON')
        nonlocal reg_method
        reg_methods = ['MLP', 'CNN', 'Linear', 'Identity']
        for reg_method in reg_methods:
            print('\nMethod:', reg_method)
            dict_data = infer_ml_data(input_dataname, output_dataname, 
                train_ind_list, bc_ind_list, unlab_ind_list,
                reg_method)
            metric = cost_metric(dict_data)
            plt.plot(metric, lw=0.75, label=reg_method)
        
        plt.xlabel(r'Time Extent, $\tau$')
        plt.ylabel('Cost')
        plt.title(rf'Prediction cost for $({mi1}, {mi2}) \rightarrow ({mo1}, {mo2})$')
        plt.legend(frameon=False, loc='best', prop={'size': 12})
        save_plot(fig, path=f'{args.results_dir}/plots/', filename='compare_metric_vs_tau')
    
    compare_ml_vs_identity()

    # NOISE TO SIGNALS
    correlator_nts_vs_nsrc()
    fit_params_nts()  # TODO
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--seed', type=int, default=42)
    add('--hdf5_filename', type=str, 
        default='../data/l64192a_run2_810-6996_1028cfgs.hdf5')
    add('--input_dataname', type=str)
    add('--output_dataname', type=str)
    add('--lr', type=float, default=0.01)
    add('--l2_coeff', type=float, default=1e-2)
    add('--training_steps', type=int, default=500) 
    add('--results_dir', type=str)

    args = parser.parse_args()
    
    with open(args.results_dir + '/data/commandline_args.dat', 'w') as f:
        args_dict = copy.deepcopy(args.__dict__)
        json.dump(args_dict, f, indent=2)

    main(args)

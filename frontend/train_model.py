"""Script to train a regression model and save its trained parameters."""
#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import argparse
import json
import copy

import sys
sys.path.insert(0, '../src/')
from utils import set_np_seed, save_model, save_data, set_plot_preferences
from processing.io_utils import get_corrs, preprocess_data, rotate_sourcetimes
from regression.torch_regressors import *
from regression.plotting import (
    plot_loss,
    plot_correlations,
    plot_correlation_heatmaps, plot_final_correlation_heatmap,
    plot_final_diag_correlations
)
from regression.utils import adjust_learning_rate, l2_regularization

from typing import TypeVar, Union

TorchRegressor = TypeVar('TorchRegressor')
SklearnRegressor = TypeVar('SklearnRegressor')


# =============================================================================
TORCH_REGRESSORS: dict[str, TorchRegressor] = {
    'Linear': LinearModel,
    'MLP': MLP,
    'CNN': CNN,
    'Transformer': Transformer,
    'Identity': torch.nn.Identity
}

SKLEARN_REGRESSORS: dict[str, SklearnRegressor] = {
    'DTR': DecisionTreeRegressor,
    'RFR': RandomForestRegressor,
    'GBR': GradientBoostingRegressor,
    'LinearRegression': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso
}

NCFG: int = 1028
NTAU: int = 192
NSRC: int = 24

if __name__ == '__main__':
    print('TRAINING ML MODEL \n data dimensions:')
    print('\t Number of time extents:', NTAU)
    print('\t Number of source times:', NSRC)
    print('\t Number of configurations:', NCFG)

    set_plot_preferences()


# =============================================================================
#  DATA PREPARATION
# =============================================================================
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


# =============================================================================
#  MODEL PREPARATION
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
    
    elif reg_method in SKLEARN_REGRESSORS.keys():
        print(f'Using {reg_method} for regression.')
        if reg_method == 'LinearRegression':
            model = LinearRegression()
        elif reg_method == 'Ridge':
            model = Ridge()
        elif reg_method == 'Lasso':
            model = Lasso()
        elif reg_method == 'DTR':
            model = DecisionTreeRegressor()
        elif reg_method == 'RFR':
            model = RandomForestRegressor()
        elif reg_method == 'GBR':
            gbr_list: list[SklearnRegressor] = []
            for _ in range(NTAU):
                gbr = GradientBoostingRegressor(
                    learning_rate = 0.1, 
                    n_estimators = 100, 
                    max_depth = 3
                )
                gbr_list.append(gbr)
            return gbr_list
    else:
        raise KeyError(f"Unknown regression method '{reg_method}'.")
    return model
        
    
# =============================================================================
#  MODEL TRAINING
# =============================================================================
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
    results_dir: str,
    track_corrs: bool
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
    correlations = []

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
        
        if track_corrs:
            prediction = prediction.detach().numpy()
            truth = output_data.detach().numpy()
            correlation = np.corrcoef(prediction, truth, rowvar=False)
            correlations.append(correlation)
    correlations = np.array(correlations)
    tau_1, tau_2 = 4, 12
    rho_diag = correlations[-1][tau_1, NTAU - 1 + tau_1]
    rho_off_diag = correlations[-1][tau_1, NTAU - 1 + tau_2]
    print('FINAL DIAG TRAIN CORRELATION B/W PRED AND TRUTH:', rho_diag)
    print('FINAL OFF-DIAG TRAIN CORRELATION B/W PRED AND TRUTH:', rho_off_diag)
    
    #input = input_data.detach().numpy()
    #truth = output_data.detach.numpy()
    #rho_base = np.corrcoef(input, truth, rowvar=False)[tau_1, NTAU - 1 + tau_2]
    #print('BASELINE CORRELATION B/W INPUT AND TRUTH:', rho_base)
    
    # Plot loss
    plot_loss(losses, results_dir)

    # Visualize correlations
    if track_corrs:
        plot_correlations(correlations, results_dir, tau_1=4)
        plot_correlations(correlations, results_dir, tau_1=4, tau_2=12)
        plot_correlation_heatmaps(correlations, results_dir)
        plot_final_correlation_heatmap(correlations, results_dir)
        plot_final_diag_correlations(correlations, results_dir)
    
    return model


def train_sklearn_model(
    input_data: torch.Tensor, 
    output_data: torch.Tensor,
    model: Union[SklearnRegressor, list[SklearnRegressor]],
) -> Union[SklearnRegressor, list[SklearnRegressor]]:
    """
    Trains the SkLearn model.

    For SkLearn models, training is done natively; however, for the 
    gradient-boosted trees, an ensemble is trained iteratively, yielding a 
    list of models for each euclidean time extent.

    Args:
        input_data: Input training data
        output_data: Output training data
        model: Regressor(s) to be trained
    
    Returns:
        Trained model
    """
    if isinstance(model, list):  # should only be the GBR
        gbr_list: list[SklearnRegressor] = []
        for tau, gbr in enumerate(model):
            print('fitting gbr at tau =', tau)
            gbr.fit(
                input_data.numpy(), 
                output_data.numpy()[:, tau]
            )
            gbr_list.append(gbr)
        return gbr_list
    else:
        model.fit(input_data.numpy(), output_data.numpy())
    return model


# =============================================================================
def main(args):
    set_np_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    global NSRC
    corr_i, corr_o = get_corrs(
        args.hdf5_filename,
        [args.input_dataname, args.output_dataname],
        NSRC
    )

    corr_i = rotate_sourcetimes(corr_i, shift=7)
    corr_o = rotate_sourcetimes(corr_o, shift=7)
    num_tau, num_cfgs, num_tsrc = corr_i.shape
    print('num_tau =', num_tau)
    print('num_cfgs =', num_cfgs)
    print('num_tsrc =', num_tsrc)

    train_ind_list = args.train_ind_list
    bc_ind_list = args.bc_ind_list
    labeled_ind_list = np.sort(train_ind_list + bc_ind_list).tolist()
    
    unlab_ind_list = []
    for i in range(num_tsrc):
        if i not in labeled_ind_list:
            unlab_ind_list.append(i)
    
    dict_data = preprocess_data(
        corr_i, corr_o,
        train_ind_list,
        bc_ind_list,
        unlab_ind_list
    )
    save_data(dict_data, path=args.results_dir + '/dict_data')
    inputs, outputs = prepare_data(dict_data)
    
    model = make_model(args.reg_method, args.seed)

    if args.reg_method != 'Identity':
        if isinstance(model, torch.nn.Module):
            trained_model = train_torch_network(
                inputs, outputs,
                lr = args.lr,
                l2_coeff = args.l2_coeff,
                training_steps = args.training_steps,
                model = model,
                results_dir = args.results_dir,
                track_corrs = args.track_corrs
            )
        else:
            trained_model = train_sklearn_model(
                inputs, outputs,
                model = model
            )
        save_model(model=trained_model, path=args.results_dir + '/model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--seed', type=int, default=42)
    add('--hdf5_filename', type=str, 
        default='../data/l64192a_run2_810-6996_1028cfgs.hdf5')
    add('--input_dataname', type=str)
    add('--output_dataname', type=str)
    add('--train_ind_list', type=str, default='[0]')
    add('--bc_ind_list', type=str, default='[3, 6, 12, 15, 18]')
    add('--reg_method', type=str, default='MLP')
    add('--lr', type=float, default=0.01)
    add('--l2_coeff', type=float, default=1e-2)
    add('--training_steps', type=int, default=500)    
    add('--track_corrs', type=int, default=1)
    add('--results_dir', type=str)

    args = parser.parse_args()
    with open(args.results_dir + '/data/commandline_args.dat', 'w') as f:
        args_dict = copy.deepcopy(args.__dict__)
        json.dump(args_dict, f, indent=2)

    args = vars(args)
    for key in ['train_ind_list', 'bc_ind_list']:
        if key in args.keys():
            args[key] = eval(args[key])
    args = argparse.Namespace(**args)
    main(args)

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
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../src/')
from utils import save_plot, save_model, set_np_seed
from regression.torch_regressors import *
from regression.utils import adjust_learning_rate, l2_regularization

from typing import TypeVar, Union

TorchRegressor = TypeVar('TorchRegressor')
SklearnRegressor = TypeVar('SklearnRegressor')


TORCH_REGRESSORS: dict[str, TorchRegressor] = {
    'Linear': LinearModel,
    'MLP': MLP,
    'CNN': CNN,
    'Transformer': Transformer
}

SKLEARN_REGRESSORS: dict[str, SklearnRegressor] = {
    'DTR': DecisionTreeRegressor,
    'RFR': RandomForestRegressor,
    'GBR': GradientBoostingRegressor,
    'LinearRegression': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso
}

NTAU = 192  # 192 time extents in our datasets


#==============================================================================
# DATA PREPARATION
#==============================================================================
def prepare_data(
    dict_data: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares the standardized two-point strange (input) and light (output)
    correlator data for input to a model for training.

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


#==============================================================================
# MODEL PREPARATION
#==============================================================================
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
        raise NotImplementedError(f"Unknown regression method '{reg_method}'.")
    return model
        
    
#==============================================================================
# MODEL TRAINING
#==============================================================================
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
    Trains the model.
    
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
    diag_correlations = []
    off_diag_correlations = []

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
            
            diag_correlations.append(correlation[4, 191+4])
            off_diag_correlations.append(correlation[4, 191+12])
    correlations = np.array(correlations)
    
    # Plot loss
    fig = plt.figure(figsize=(8., 6.))
    plt.plot(losses, color='firebrick')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='training_loss')

    if track_corrs:
        fig = plt.figure(figsize=(8., 6.))
        for tau in range(1, 6):
            plt.plot(correlations[:, tau, 191 + tau], label=rf'$\tau={tau}$')
        plt.hlines(1.0, 0, training_steps, color='black', linestyle='dashed')
        plt.ylabel(r"$\rho(O(\tau), O^{\mathrm{pred}}(\tau))$")
        plt.xlabel('Training Iterations')
        plt.legend()
        save_plot(fig=fig, path=f'{results_dir}/plots/', filename='diag_training_correlation')

        fig = plt.figure(figsize=(8., 6.))
        for tau in range(1, 20):
            plt.plot(correlations[:, tau, 191 + 12], label=rf'$\tau={tau}$')
            plt.plot(correlations[:, 191 + tau, 191 + 12], label=rf'Truth, $\tau={tau}$', linestyle='dashed')
        #plt.plot(correlations[:, 191 + 12, 191 + tau], c='k')
        plt.ylabel(r"$\rho(O(\tau'=12), O^{\mathrm{pred}}(\tau))$")
        plt.xlabel('Training Iterations')
        plt.legend()
        save_plot(fig=fig, path=f'{results_dir}/plots/', filename='off_diag_training_correlation')

        # Save plots of correlation heatmaps over training time
        fig, axes = plt.subplots(1, 5, sharey=True, figsize=(20, 4.))
        fig.supylabel(r"$\rho(O(\tau), O^{\mathrm{pred}}(\tau'))$")
        for i in range(4):
            ax = axes[i]
            im = ax.imshow(correlations[50*i], cmap='hot')
            im.norm.autoscale([0, 1])
            ax.set_xlabel(f'Iter {50*i}')
        im = axes[-1].imshow(correlations[-1], cmap='hot')
        im.norm.autoscale([0, 1])
        axes[-1].set_xlabel(f'Iter {len(losses)}')
        #cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.])
        #fig.colorbar(im, cax=cbar_ax)
        save_plot(fig=fig, path=f'{results_dir}/plots/', filename='correlation_heatmaps')

        fig = plt.figure(figsize=(8., 8.))
        plt.title(r"$\rho(O(\tau), O^{\mathrm{pred}}(\tau'))$")
        plt.imshow(correlations[-1], cmap='hot')
        save_plot(fig=fig, path=f'{results_dir}/plots/', filename='final_correlation')
    return model


def train_sklearn_model(
    input_data: torch.Tensor, 
    output_data: torch.Tensor,
    model: Union[SklearnRegressor, list[SklearnRegressor]],
) -> Union[SklearnRegressor, list[SklearnRegressor]]:
    """
    Trains the model.

    For the SkLearn models, training is done natively; however, for the 
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


#==============================================================================
def main(args):
    seed = args.seed
    set_np_seed(seed)
    torch.set_default_dtype(torch.float64)

    inputs, outputs = prepare_data(dict_data)
    
    model = make_model(args.reg_method, args.seed)
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
    add('--reg_method', type=str, default='MLP')
    add('--lr', type=float, default=0.01)
    add('--l2_coeff', type=float, default=1e-2)
    add('--training_steps', type=int, default=500)    
    add('--track_corrs', type=bool, default=1)
    add('--results_dir', type=str)

    args = parser.parse_args()

    with open(args.results_dir+'/data/commandline_args.dat', 'w') as f:
        args_dict = copy.deepcopy(args.__dict__)
        args_dict['dict_hyperparams'] = json.loads(args.dict_hyperparams)
        json.dump(args_dict, f, indent=2)

    main(args)

import torch
import torch.nn.functional as F

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import argparse
import json
import matplotlib.pyplot as plt

from .torch_regressors import *
from .utils import adjust_learning_rate
import sys
sys.path.insert(0, '../')
from utils import save_plot

from typing import TypeVar, Union, List

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
    'Linear': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso
}

NTAU = 192  # 192 time extents in our datasets


#===================================================================================================
# MODEL PREPARATION AND TRAINING
#===================================================================================================
def make_model(
    args: argparse.Namespace
) -> Union[TorchRegressor, SklearnRegressor, List[SklearnRegressor]]:
    """
    Prepares a regression model to be trained.

    Args:
        args: A Namespace object containing arguments

    Returns:
        model: An initialized PyTorch or SkLearn regressor
    """
    torch.manual_seed(args.seed)
    use_torch: bool = args.use_torch
    reg_method: str = args.reg_method

    global NTAU

    if isinstance(TORCH_REGRESSORS[reg_method], torch.nn.Module):
        assert use_torch, 'Torch regressor supplied but not using torch.'
        print(f'Using {reg_method} for regression.')

        if reg_method == 'MLP':
            model = MLP(NTAU, NTAU, hiddden_dims=[NTAU // 4], batch_norm = True)
        elif reg_method == 'Linear':
            model = LinearModel(NTAU, NTAU)
        elif reg_method == 'CNN':
            model = CNN(NTAU, NTAU, hidden_channels=[1], kernel_size=15, batch_norm=False)
        elif reg_method == 'Transformer':
            model = Transformer(input_dim=1, num_heads=1)
        else:
            raise NotImplementedError(f'Unknown Torch regression method {reg_method}.')
    else:
        print(f'Using {reg_method} for regression.')

        if reg_method == 'Linear':
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
            gbr_list: List[SklearnRegressor] = []
            for _ in range(NTAU):
                gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
                gbr_list.append(gbr)
            return gbr_list
        else:
            raise NotImplementedError(f'Unknown regression method {reg_method}.')
    return model
        
    
def train_model(
    dict_data: dict[str, torch.Tensor], 
    args: argparse.Namespace,
    model: Union[TorchRegressor, SklearnRegressor, List[SklearnRegressor]]
) -> Union[TorchRegressor, SklearnRegressor, List[SklearnRegressor]]:
    """
    Trains the model.
    
    For neural networks implemented via PyTorch, training is done according to MSE loss with 
    :math:`\ell^2` regularization. The training loss curve is saved and plotted.

    For the SkLearn models, training is done natively; however, for the gradient-boosted trees, an
    ensemble is trained iteratively, yielding a list of models for each euclidean time extent.

    Args:
        dict_data: Dictionary of preprocessed correlator data
        args: A Namespace object containing arguments
    
    Returns:
        Trained model
    """
    corr_i_train_tensor = dict_data["corr_i_train_tensor"]
    corr_o_train_tensor = dict_data["corr_o_train_tensor"]
    corr_i_train_means = dict_data["corr_i_train_means"]
    corr_o_train_means = dict_data["corr_o_train_means"]
    corr_i_train_stds = dict_data["corr_i_train_stds"]
    corr_o_train_stds = dict_data["corr_o_train_stds"]

    n_corr_2pt_s_train_tensor = (corr_i_train_tensor - corr_i_train_means) / corr_i_train_stds
    n_corr_2pt_l_train_tensor = (corr_o_train_tensor - corr_o_train_means) / corr_o_train_stds
    
    dict_hyperparams = json.loads(args.dict_hyperparams)
    lr = dict_hyperparams['lr']
    l2_coeff = dict_hyperparams['l2_coeff']
    training_steps = dict_hyperparams['training_steps']

    # Pytorch workflow for training neural net
    if isinstance(model, torch.nn.Module):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []

        for i in range(training_steps):
            lr2 = adjust_learning_rate(training_steps, 0.3, lr, optimizer, i)

            prediction = model(n_corr_2pt_s_train_tensor)
            loss = F.mse_loss(prediction, n_corr_2pt_l_train_tensor)

            l2_regularization = l2_coeff * sum([(p**2).sum() for (_, p) in model.named_parameters()])
            loss = loss + l2_regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Step:', i)
                print(f'Loss: {loss.item():.12f} | lr: {lr2:.12f}')
            losses.append(loss.item())
        # Plot loss
        fig = plt.figure(figsize=(8., 6.))
        plt.plot(losses, c='k')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        save_plot(fig=fig, path='../../plots', filename='training_loss')
    
    # Sklearn regressor training
    else:
        if hasattr(model, '__iter__'):  # should only be the gradient-boosted trees
            def fit_gbr(gbr_list: List[SklearnRegressor]) -> List[SklearnRegressor]:
                """Each gbr is fit to a single time extent in the target correlator."""
                nonlocal model
                model = []
                for tau, gbr in enumerate(gbr_list):
                    gbr.fit(
                        n_corr_2pt_s_train_tensor.numpy(), 
                        n_corr_2pt_l_train_tensor.numpy()[:, tau]
                    )
                    model.append(gbr)
                return model
            model = map(fit_gbr, model)
        else:
            model.fit(n_corr_2pt_s_train_tensor.numpy(), n_corr_2pt_l_train_tensor.numpy())
    return model

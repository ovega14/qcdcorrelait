import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../')
from utils import save_plot

from typing import TypeVar, Union, List

TorchRegressor = TypeVar('TorchRegressor')
SklearnRegressor = TypeVar('SklearnRegressor')


# =============================================================================
#  PREDICTION
# =============================================================================
@torch.no_grad()
def predict(
    n_corr_i_train_tensor: torch.Tensor,
    model: Union[TorchRegressor, SklearnRegressor, List[SklearnRegressor]],
    reg_method: str,
    dict_data: dict[str, torch.tensor]
) -> dict[str, torch.Tensor]:
    """
    Uses the trained regressor to infer new output data from given input data.

    Args:
        n_corr_i_train_tensor: The unlabeled input correlator data from which 
            to predict
        model: A trained regressor
        reg_method: Name of regression method being used
        dict_data: Dictionary of preprocessed data
    
    Returns:
        Dictionary of predicted correlator data.
    """
    corr_i_unlab_tensor = dict_data["corr_i_unlab_tensor"]
    corr_i_train_means = dict_data["corr_i_train_means"]
    corr_o_train_means = dict_data["corr_o_train_means"]
    corr_i_train_stds = dict_data["corr_i_train_stds"]
    corr_o_train_stds = dict_data["corr_o_train_stds"]
    
    n_corr_i_unlab_tensor = (corr_i_unlab_tensor - corr_i_train_means) / corr_i_train_stds
    
    corr_i_bc_tensor = dict_data["corr_i_bc_tensor"]
    n_corr_i_bc_tensor = (corr_i_bc_tensor - corr_i_train_means) / corr_i_train_stds
    
    if reg_method == 'Identity':
        n_corr_o_train_pred_tensor = copy.deepcopy(n_corr_i_train_tensor)
        n_corr_o_unlab_pred_tensor = copy.deepcopy(n_corr_i_unlab_tensor)
        n_corr_o_bc_pred_tensor = copy.deepcopy(n_corr_i_bc_tensor)
    elif isinstance(model, torch.nn.Module):
        net = model
        net.eval()
        with torch.no_grad():
            n_corr_o_train_pred_tensor = net(copy.deepcopy(n_corr_i_train_tensor))
            n_corr_o_unlab_pred_tensor = net(copy.deepcopy(n_corr_i_unlab_tensor))
            n_corr_o_bc_pred_tensor = net(copy.deepcopy(n_corr_i_bc_tensor))
    else:
        if reg_method == 'GBR':
            n_corr_o_train_pred_tensor = []
            n_corr_o_unlab_pred_tensor = []
            n_corr_o_bc_pred_tensor = []
            
            gbr_list = model
            for reg in gbr_list:  # one for each tau
                train_pred = reg.predict(n_corr_i_train_tensor.numpy())
                unlab_pred = reg.predict(n_corr_i_unlab_tensor.numpy())
                bc_pred = reg.predict(n_corr_i_bc_tensor.numpy())

                n_corr_o_train_pred_tensor.append(train_pred)
                n_corr_o_unlab_pred_tensor.append(unlab_pred)
                n_corr_o_bc_pred_tensor.append(bc_pred)

            n_corr_o_train_pred_tensor = torch.from_numpy(np.array(n_corr_o_train_pred_tensor)).T
            n_corr_o_unlab_pred_tensor = torch.from_numpy(np.array(n_corr_o_unlab_pred_tensor)).T
            n_corr_o_bc_pred_tensor = torch.from_numpy(np.array(n_corr_o_bc_pred_tensor)).T
        else:
            reg = model
            n_corr_o_train_pred_tensor = torch.from_numpy(reg.predict(n_corr_i_train_tensor.numpy()))
            n_corr_o_unlab_pred_tensor = torch.from_numpy(reg.predict(n_corr_i_unlab_tensor.numpy()))
            n_corr_o_bc_pred_tensor = torch.from_numpy(reg.predict(n_corr_i_bc_tensor.numpy()))
    
    # Denormalize predicted correlators
    corr_o_train_pred_tensor = n_corr_o_train_pred_tensor * corr_o_train_stds + corr_o_train_means
    corr_o_unlab_pred_tensor = n_corr_o_unlab_pred_tensor * corr_o_train_stds + corr_o_train_means
    corr_o_bc_pred_tensor = n_corr_o_bc_pred_tensor * corr_o_train_stds + corr_o_train_means

    dict_results = {}
    dict_results["corr_o_train_pred_tensor"] = corr_o_train_pred_tensor
    dict_results["corr_o_bc_pred_tensor"] = corr_o_bc_pred_tensor
    dict_results["corr_o_unlab_pred_tensor"] = corr_o_unlab_pred_tensor

    dict_results["n_corr_o_train_pred_tensor"] = n_corr_o_train_pred_tensor
    dict_results["n_corr_o_bc_pred_tensor"] = n_corr_o_bc_pred_tensor
    dict_results["n_corr_o_unlab_pred_tensor"] = n_corr_o_unlab_pred_tensor
    
    return dict_results


# =============================================================================
#  VISUALIZATION OF FEATURE IMPORTANCE AND INTERPRETABILITY
# =============================================================================
def get_feature_importances(model: SklearnRegressor) -> None:
    feature_importances = model.feature_importances_
    
    fig = plt.figure(figsize=(8., 8.))
    plt.scatter(np.array(range(len(feature_importances))), feature_importances)
    plt.xlabel(r'Time extent, $\tau$')
    plt.ylabel('Importance')
    save_plot(fig=fig, path='plots/', filename='feature_importances')

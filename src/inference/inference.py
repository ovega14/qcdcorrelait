import torch
import numpy as np
import argparse
import copy

from .. plotting import save_plot

from typing import TypeVar, Union, List

TorchRegressor = TypeVar('TorchRegressor')
SklearnRegressor = TypeVar('SklearnRegressor')


#===================================================================================================
# PREDICTION
#===================================================================================================
def predict(
    n_corr_i_train_tensor: torch.Tensor,
    model: Union[TorchRegressor, SklearnRegressor, List[SklearnRegressor]],
    args: argparse.Namespace,
    dict_data: dict[str, torch.tensor]
) -> dict[str, torch.Tensor]:
    """
    Uses the trained regressor to predict new output data from given input data.

    Args:
        n_corr_i_train_tensor: The unlabeled input correlator data from which to predict
        model: A trained regressor
        args: A namespace object containing arguments
        dict_data: Dictionary of preprocessed data
    
    Returns:
        Dictionary of predicted correlator data.
    """
    corr_i_unlab_tensor = dict_data["corr_i_unlab_tensor"]
    n_corr_i_unlab_tensor = (corr_i_unlab_tensor - corr_i_train_means) / corr_i_train_stds
    
    corr_i_bc_tensor = dict_data["corr_i_bc_tensor"]
    n_corr_i_bc_tensor = (corr_i_bc_tensor - corr_i_train_means) / corr_i_train_stds

    corr_i_train_means = dict_data["corr_i_train_means"]
    corr_o_train_means = dict_data["corr_o_train_means"]
    corr_i_train_stds = dict_data["corr_i_train_stds"]
    corr_o_train_stds = dict_data["corr_o_train_stds"]
    
    use_torch: bool = args.use_torch
    reg_method = args.reg_method
    
    if use_torch == 1:
        net = model
        net.eval()
        with torch.no_grad():
            n_corr_o_train_pred_tensor = net(copy.deepcopy(n_corr_i_train_tensor))
            n_corr_o_unlab_pred_tensor = net(copy.deepcopy(n_corr_i_unlab_tensor))
            n_corr_o_bc_pred_tensor = net(copy.deepcopy(n_corr_i_bc_tensor))
    else:
        if reg_method == 'GBR':
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


#===================================================================================================
# VISUALIZATION OF FEATURE IMPORTANCE AND INTERPRETABILITY
#===================================================================================================

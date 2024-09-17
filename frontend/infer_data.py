"""Script to infer and save new correlator data."""
#!/usr/bin/env python3
import torch
import numpy as np
import numpy.typing as npt

import argparse
import copy
import json

import sys
sys.path.insert(0, '../src/')
from processing.io_utils import get_corrs
from processing.conversion import tensor_to_avg_over_tsrc
from regression.inference import predict
from regression.ratio_method import RatioMethod
from analysis.plotting import (
    plot_correlators,
    plot_relative_correlated_difference,
    plot_noise_to_signal,
    plot_normalized_noise_to_signal,
    plot_error_breakdown
)
from utils import (
    load_model, set_np_seed, 
    save_data, load_data, 
    set_plot_preferences
)


# =============================================================================
NCFG: int = 1028
NTAU: int = 192
NSRC: int = 24

if __name__ == '__main__':
    print('INFERRING NEW CORRELATOR DATA \n data dimensions:')
    print('\t Number of time extents:', NTAU)
    print('\t Number of source times:', NSRC)
    print('\t Number of configurations:', NCFG)

    set_plot_preferences()


# =============================================================================
#  RATIO ESTIMATOR INFERENCE
# =============================================================================
def infer_ratio_method(
    corr_i: npt.NDArray, 
    corr_o: npt.NDArray, 
    corr_o_pred: npt.NDArray,
    *, 
    lab_ind_list: list[int], 
    boosted: bool
) -> tuple[npt.NDArray, npt.NDArray]:
    # Get ratio method data
    ratio_method = RatioMethod(
        corr_i = corr_i, 
        corr_o = corr_o, 
        lab_ind_list = lab_ind_list,
        boosted = boosted
    )
    ds_ratio_method = ratio_method.predict()

    ml_ratio_method = RatioMethod(
        corr_i = corr_o_pred,
        corr_o = corr_o,
        lab_ind_list = lab_ind_list,
        boosted = boosted
    )
    ds_ml_ratio_method = ml_ratio_method.predict()

    return ds_ratio_method, ds_ml_ratio_method


# =============================================================================
def main(args):
    set_np_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    global NCFG, NSRC, NTAU

    model = load_model(args.results_dir + '/model')
    dict_data = load_data(args.results_dir + '/dict_data')

    # Pre-process -------------------------------------------------------------
    corr_i, corr_o = get_corrs(
        args.hdf5_filename,
        [args.input_dataname, args.output_dataname],
        NSRC
    )

    n_corr_i_train_tensor = (dict_data["corr_i_train_tensor"] - dict_data["corr_i_train_means"]) / dict_data["corr_i_train_stds"]
    
    dict_results = predict(
        n_corr_i_train_tensor = n_corr_i_train_tensor,
        model = model,
        reg_method = args.reg_method,
        dict_data = dict_data
    )

    dict_data["n_corr_o_train_tensor"] = (dict_data["corr_o_train_tensor"] - dict_data["corr_o_train_means"]) / dict_data["corr_o_train_stds"]

    dict_data["n_corr_i_unlab_tensor"] = (dict_data["corr_i_unlab_tensor"] - dict_data["corr_i_train_means"]) / dict_data["corr_i_train_stds"]
    dict_data["n_corr_o_unlab_tensor"] = (dict_data["corr_o_unlab_tensor"] - dict_data["corr_o_train_means"]) / dict_data["corr_o_train_stds"]

    dict_data["n_corr_i_bc_tensor"] = (dict_data["corr_i_bc_tensor"] -  dict_data["corr_i_train_means"]) /  dict_data["corr_i_train_stds"]
    dict_data["n_corr_o_bc_tensor"] = (dict_data["corr_o_bc_tensor"] -  dict_data["corr_o_train_means"]) /  dict_data["corr_o_train_stds"]

    lab_ind_list = args.train_ind_list + args.bc_ind_list

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
    for i in range(NSRC):
        if i in args.train_ind_list:
            corr_o_pred.append(corr_o_train_pred[:, :, curr_ind_train])
            curr_ind_train += 1
        elif i in args.bc_ind_list:
            corr_o_pred.append(corr_o_bc_pred[:, :, curr_ind_bc])
            curr_ind_bc += 1
        else:
            corr_o_pred.append(corr_o_unlab_pred[:, :, curr_ind_unlab])
            curr_ind_unlab += 1
    corr_o_pred = np.array(corr_o_pred)
    corr_o_pred = np.swapaxes(corr_o_pred, 0, 1)
    corr_o_pred = np.swapaxes(corr_o_pred, 1, 2)
    print("corr_o_pred.shape:", corr_o_pred.shape, )
    print("corr_o.shape:", corr_o.shape, )

    # Get & save ratio method data --------------------------------------------
    ds_ratio_method, ds_ml_ratio_method = infer_ratio_method(
        corr_i, corr_o, corr_o_pred, 
        lab_ind_list=lab_ind_list, 
        boosted = args.modify_ratio)
    
    # Post-process ------------------------------------------------------------
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
    corr_o_train_truth = np.average(corr_o[:, :, args.train_ind_list], axis=-1)
    corr_o_labeled_truth = np.average(corr_o[:, :, args.train_ind_list + args.bc_ind_list], axis=-1)

    dict_data['corr_o_truth'] = corr_o_truth
    dict_data['corr_o_pred_corrected'] = corr_o_pred_corrected
    dict_data['corr_o_pred_uncorrected'] = corr_o_pred_uncorrected

    # Plot correlators & their statistics -------------------------------------
    # Compute correlation b/w test and pred data
    tau_1 = 4
    tau_2 = 12
    unlab_pred = np.average(corr_o_unlab_pred, axis=-1)
    corr_mat = np.corrcoef(unlab_pred, corr_o_truth, rowvar=True)
    print('corr_mat shape:', corr_mat.shape)  
    rho = corr_mat[tau_1, NTAU - 1 + tau_2]
    print('CORRELATION B/W TEST PREDICTED DATA AND TRUTH DATA:', rho)

    plot_correlators(
        num_tau = NTAU,
        corr_o_truth = corr_o_truth,
        corr_o_train_truth = corr_o_train_truth,
        corr_o_pred_corrected = corr_o_pred_corrected,
        corr_o_pred_uncorrected = corr_o_pred_uncorrected,
        ds_ratio_method = ds_ratio_method,
        ds_ml_ratio_method = ds_ml_ratio_method,
        results_dir = args.results_dir,
        args = args  # TODO: get rid of args as an arg for this plot func
    )

    plot_relative_correlated_difference(
        n_corr_o_unlab_vs_tau, 
        n_corr_o_unlab_pred_vs_tau,
        n_corr_o_bc_vs_tau,
        n_corr_o_bc_pred_vs_tau,
        results_dir = args.results_dir
    )

    plot_noise_to_signal(
        num_tau = NTAU,
        corr_o_truth = corr_o_truth,
        corr_o_labeled_truth = corr_o_labeled_truth,
        corr_o_pred_corrected = corr_o_pred_corrected,
        corr_o_pred_uncorrected = corr_o_pred_uncorrected,
        ds_ratio_method = ds_ratio_method,
        ds_ml_ratio_method = ds_ml_ratio_method,
        results_dir = args.results_dir,
        args = args  # TODO: get rid of args as an arg for this plot func
    )

    plot_normalized_noise_to_signal(
        num_tau = NTAU,
        corr_o_truth = corr_o_truth,
        corr_o_labeled_truth = corr_o_labeled_truth,
        corr_o_pred_corrected = corr_o_pred_corrected,
        corr_o_pred_uncorrected = corr_o_pred_uncorrected,
        ds_ratio_method = ds_ratio_method,
        ds_ml_ratio_method = ds_ml_ratio_method,
        results_dir = args.results_dir,
        args = args  # TODO: get rid of args as an arg for this plot func
    )

    plot_error_breakdown(
        pred_corrected = corr_o_pred_corrected,
        pred_uncorrected = corr_o_pred_uncorrected,
        bias_correction = corr_o_bias_correction,
        results_dir = args.results_dir,
        fig_name = 'error_breakdown',
        truth = corr_o_truth,
    )

    # Save all datasets -------------------------------------------------------
    save_data(dict_data, path=args.results_dir + '/dict_data')
    save_data(dict_results, path=args.results_dir + '/dict_results')
    save_data(ds_ratio_method, path=args.results_dir + '/ds_ratio_method')
    save_data(ds_ml_ratio_method, path=args.results_dir + '/ds_ml_ratio_method')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--seed', type=int, default=42)
    add('--hdf5_filename', type=str, 
        default='../data/l64192a_run2_810-6996_1028cfgs.hdf5')
    add('--input_dataname', type=str)
    add('--output_dataname', type=str)
    add('--reg_method', type=str, default='MLP')
    add('--compare_ratio_method', type=bool, default=1)
    add('--compare_ml_ratio_method', type=bool, default=1)
    add('--train_ind_list', type=str, default='[0]')
    add('--bc_ind_list', type=str, default='[3, 6, 12, 15, 18]')
    add('--modify_ratio', type=bool, default=1)
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

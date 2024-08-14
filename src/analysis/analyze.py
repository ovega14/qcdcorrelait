import numpy as np
import gvar as gv
import numpy.typing as npt
from typing import List, Any, TypeVar
Fit = TypeVar('Fit')

from .plot import *
from .tabulate import FitParamsTable
sys.path.insert(0, '../')
from regression.ratio_method import RatioMethod
from processing.preprocessing import tensor_to_avg_over_tsrc
from fitting.fit import fit_corrs


def analysis_pred(
    corr_i, 
    corr_o,
    num_tau: int, 
    num_cfgs: int, 
    num_tsrc: int,
    dict_data: dict[str, npt.NDArray],
    dict_results: dict[str, npt.NDArray],
    reg_method,
    results_dir,
    args,
) -> dict[str, dict[str, Fit]]:
    """
    Performs the full analysis of the predicted correlator data, including plots and tables.

    Args:
        corr_i: 
        corr_o:
        num_tau: Number of Euclidean time extents used in fitting the correlator data
        num_cfgs: Number of configurations in the original dataset
        num_tsrc: Number of source times in the original dataset
        dict_data: Dictionary of originally preprocessed correlator data
        dict_results: Dictionary of resultant correlator data from inference
    """
    # COLLECT NECESSSARY DATA FROM INPUTS AND RESULTS ----------------------------------------------
    n_corr_o_unlab_tensor = dict_data["n_corr_o_unlab_tensor"]
    n_corr_o_bc_tensor = dict_data["n_corr_o_bc_tensor"]
    corr_o_train_stds = dict_data["corr_o_train_stds"]
    corr_o_train_means = dict_data["corr_o_train_means"]

    corr_o_train_pred_tensor = dict_results["corr_o_train_pred_tensor"]
    corr_o_unlab_pred_tensor = dict_results["corr_o_unlab_pred_tensor"]
    corr_o_bc_pred_tensor = dict_results["corr_o_bc_pred_tensor"]
    n_corr_o_unlab_pred_tensor = dict_results["n_corr_o_unlab_pred_tensor"]
    n_corr_o_bc_pred_tensor = dict_results["n_corr_o_bc_pred_tensor"]

    # Get ratio method data
    ratio_method = RatioMethod(
        corr_i = corr_i, 
        corr_o = corr_o, 
        lab_ind_list = args.train_ind_list + args.bc_ind_list,
        boosted = args.modify_ratio
    )
    ds_ratio_method = ratio_method.predict()

    corr_o_train_pred = corr_o_train_pred_tensor.T.reshape((num_tau, num_cfgs, -1)).numpy()
    corr_o_bc_pred = corr_o_bc_pred_tensor.T.reshape((num_tau, num_cfgs, -1)).numpy()
    corr_o_unlab_pred = corr_o_unlab_pred_tensor.T.reshape((num_tau, num_cfgs, -1)).numpy()
    corr_o_pred = []
    curr_ind_train = 0
    curr_ind_bc = 0
    curr_ind_unlab = 0
    for i in range(num_tsrc):
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
    ml_ratio_method = RatioMethod(
        corr_i = corr_o_pred,
        corr_o = corr_o,
        lab_ind_list = args.train_ind_list + args.bc_ind_list,
        boosted = args.modify_ratio
    )
    ds_ml_ratio_method = ml_ratio_method.predict()

    # POST-PROCESS DATA ----------------------------------------------------------------------------
    n_corr_o_unlab_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_unlab_tensor, num_tau, num_cfgs)
    n_corr_o_unlab_pred_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_unlab_pred_tensor, num_tau, num_cfgs)

    n_corr_o_bc_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_bc_tensor, num_tau, num_cfgs)
    n_corr_o_bc_pred_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_bc_pred_tensor, num_tau, num_cfgs)

    n_corr_o_uncorrected = n_corr_o_unlab_pred_vs_tau
    n_corr_o_corrected = n_corr_o_uncorrected + n_corr_o_bc_vs_tau - n_corr_o_bc_pred_vs_tau
    n_bias_correction_vs_tau = n_corr_o_bc_vs_tau - n_corr_o_bc_pred_vs_tau

    corr_o_pred_corrected = n_corr_o_corrected * corr_o_train_stds[:, None] + corr_o_train_means[:, None]
    corr_o_pred_uncorrected = n_corr_o_uncorrected * corr_o_train_stds[:, np.newaxis] + corr_o_train_means[:, np.newaxis]
    corr_o_bias_correction = n_bias_correction_vs_tau * corr_o_train_stds[:, np.newaxis] + corr_o_train_means[:, np.newaxis]
    
    corr_o_truth = np.average(corr_o, axis=-1)
    corr_o_train_truth = np.average(corr_o[:, :, args.train_ind_list], axis=-1)
    corr_o_labeled_truth = np.average(corr_o[:, :, args.train_ind_list + args.bc_ind_list], axis=-1)

    # MAKE PLOTS -----------------------------------------------------------------------------------
    plot_correlators(
        num_tau = num_tau,
        corr_o_truth = corr_o_truth,
        corr_o_train_truth = corr_o_train_truth,
        corr_o_pred_corrected = corr_o_pred_corrected,
        corr_o_pred_uncorrected = corr_o_pred_uncorrected,
        ds_ratio_method = ds_ratio_method,
        ds_ml_ratio_method = ds_ml_ratio_method,
        results_dir = results_dir,
        args = args
    )

    plot_relative_correlated_difference(
        n_corr_o_unlab_vs_tau, 
        n_corr_o_unlab_pred_vs_tau,
        n_corr_o_bc_vs_tau,
        n_corr_o_bc_pred_vs_tau,
        results_dir = results_dir
    )

    plot_noise_to_signal(
        num_tau = num_tau,
        corr_o_truth = corr_o_truth,
        corr_o_labeled_truth = corr_o_labeled_truth,
        corr_o_pred_corrected = corr_o_pred_corrected,
        corr_o_pred_uncorrected = corr_o_pred_uncorrected,
        ds_ratio_method = ds_ratio_method,
        ds_ml_ratio_method = ds_ml_ratio_method,
        results_dir = results_dir,
        args = args
    )

    plot_normalized_noise_to_signal(
        num_tau = num_tau,
        corr_o_truth = corr_o_truth,
        corr_o_labeled_truth = corr_o_labeled_truth,
        corr_o_pred_corrected = corr_o_pred_corrected,
        corr_o_pred_uncorrected = corr_o_pred_uncorrected,
        ds_ratio_method = ds_ratio_method,
        ds_ml_ratio_method = ds_ml_ratio_method,
        results_dir = results_dir,
        args = args
    )

    plot_error_breakdown(
        pred_corrected = corr_o_pred_corrected,
        pred_uncorrected = corr_o_pred_uncorrected,
        bias_correction = corr_o_bias_correction,
        results_dir = results_dir,
        fig_name = 'error_breakdown',
        truth = corr_o_truth,
    )

    # FIT CORRELATORS-------------------------------------------------------------------------------
    dict_orig_corrs = dict()
    corrs = [corr_o_truth, corr_o_pred_corrected, corr_o_pred_uncorrected]
    names = ["corr_o_truth", "corr_o_pred_corrected", "corr_o_pred_uncorrected"]
    for name, corr in zip(names, corrs):
        dict_orig_corrs[name] = corr

    dict_opts = dict()
    opt_keys: List[str] = [
        'filename', 
        'tp', 
        'tmin', 
        'tmax', 
        'ne', 
        'no', 
        'maxit', 
        'averages_tsrc'
    ]
    opt_vals: List[Any] = [
        args.input_dataname + '_' + args.output_dataname, 
        num_tau, 
        2, 
        num_tau-2, 
        5, 
        5, 
        5_000, 
        False
    ]
    for key, val in zip(opt_keys, opt_vals):
        dict_opts[key] = val
    
    filename = dict_opts.get('filename')
    dict_fits = fit_corrs(dict_orig_corrs, dict_opts)
    all_dict_fits = dict_fits.copy()

    # WRITE RESULTS---------------------------------------------------------------------------------
    with open(results_dir + '/results/fits.txt', 'w') as f:
        for tag in dict_fits.keys():
            print(tag, file=f)
            print(dict_fits[tag], file=f)
    with open(results_dir + '/results/latex_table.txt', 'w') as f:
        for tag in dict_fits.keys():
            print(tag + ':\n', file=f)
            print(FitParamsTable.write_line(reg_method, dict_fits, filename, tag), file=f)
            print('=' * 120, file=f)

    if args.compare_ratio_method == 1:
        dict_fits = fit_corrs(
            dict_corrs = None,
            dict_opts = dict_opts,
            gv_ds = ds_ratio_method,
            excluding_tags=['hp_i', 'lp_i']
        )
        with open(results_dir + '/results/fits.txt', 'a') as f:
            for tag in dict_fits.keys():
                print(tag, file=f)
                print(dict_fits[tag], file=f)
        with open(results_dir + '/results/latex_table.txt', 'a') as f:
            for tag in dict_fits.keys():
                if tag == 'ratio_method_pred':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line('RM', dict_fits, filename, tag), file=f)
                elif tag == 'ratio_method_pred_modified':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line('bRM', dict_fits, filename, tag), file=f)
                else:
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(reg_method, dict_fits, filename, tag), file=f)
                print('=' * 120, file=f)
        all_dict_fits.update(dict_fits)
    
    if args.compare_ml_ratio_method == 1:
        dict_fits = fit_corrs(
            dict_corrs = None,
            dict_opts = dict_opts,
            gv_ds = ds_ml_ratio_method,
            excluding_tags=['hp_o', 'hp_i', 'lp_i', 'lp_o']
        )
        with open(results_dir + '/results/fits.txt', 'a') as f:
            for tag in dict_fits.keys():
                if tag == 'ratio_method_pred':
                    print("ml_ratio_method_pred", file=f)
                elif tag == 'ratio_method_pred_modified':
                    print("ml_ratio_method_pred_modified", file=f)
                else:
                    print(tag, file=f)
                print(dict_fits[tag], file=f)
        with open(results_dir + '/results/latex_table.txt', 'a') as f:
            for tag in dict_fits.keys():
                if tag == 'ratio_method_pred':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line('RM + ML', dict_fits, filename, tag), file=f)
                elif tag == 'ratio_method_pred_modified':
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line('bRM + ML', dict_fits, filename, tag), file=f)
                else:
                    print(tag + ':\n', file=f)
                    print(FitParamsTable.write_line(reg_method, dict_fits, filename, tag), file=f)
                print('=' * 120, file=f)
        all_dict_fits.update(dict_fits)

        return all_dict_fits

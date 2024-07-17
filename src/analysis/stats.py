import numpy as np
from scipy.stats import sem
import matplotlib
import matplotlib.pyplot as plt

import numpy.typing as npt

from ..plotting import save_plot
from ..processing.preprocessing import tensor_to_avg_over_tsrc


#===================================================================================================
# PLOTTING PREFERENCES
#===================================================================================================
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
font = {
    'weight' : 'normal', 
    'size': 18
}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # for \text command


#===================================================================================================
# CORRELATORS
#===================================================================================================
def plot_correlators(*args) -> None:
    """
    Plots correlator data along with predicted correlators on the same figure.
    """
    corr_o_pred_corrected = n_corr_o_corrected * corr_o_train_stds[:, None] + corr_o_train_means[:, None]
    corr_o_pred_uncorrected = n_corr_o_uncorrected * corr_o_train_stds[:, np.newaxis] + corr_o_train_means[:, np.newaxis]
    corr_o_bias_correction = n_bias_correction_vs_tau * corr_o_train_stds[:, np.newaxis] + corr_o_train_means[:, np.newaxis]
    corr_o_truth = np.average(corr_o, axis=-1)
    corr_o_train_truth = np.average(corr_o[:, :, args.train_ind_list], axis=-1)
    corr_o_labeled_truth = np.average(corr_o[:, :, args.train_ind_list + args.bc_ind_list], axis=-1)

    fig = plt.figure(figsize=(8., 6.))

    plt.errorbar(
        x=np.arange(0, n_tau),
        y=np.average(corr_o_truth, axis=-1),
        yerr=sem(corr_o_truth, axis=-1),
        c='b',
        label='Truth',
    )

    plt.errorbar(
        x=np.arange(0, n_tau),
        y=np.average(corr_o_train_truth, axis=-1),
        yerr=sem(corr_o_train_truth, axis=-1),
        c='g',
        label='Training set',
    )

    plt.errorbar(
        x=np.arange(0, n_tau),
        y=np.average(corr_o_pred_corrected, axis=-1),
        yerr=sem(corr_o_pred_corrected, axis=-1),
        c='r',
        label='Pred Corrected',
    )

    plt.errorbar(
        x=np.arange(0, n_tau),
        y=np.average(corr_o_pred_uncorrected, axis=-1),
        yerr=sem(corr_o_pred_uncorrected, axis=-1),
        c='gold',
        label='Pred Uncorrected',
    )

    if args.compare_ratio_method == 1:
        ds_ratio_method = ratio_method(
            corr_i=corr_i,
            corr_o=corr_o,
            lab_ind_list=args.train_ind_list+args.bc_ind_list,
            modify=args.modify_ratio,
        )

    if args.compare_ml_ratio_method == 1:
        corr_o_train_pred = corr_o_train_pred_tensor.T.reshape((n_tau, n_configs, -1)).numpy()
        corr_o_bc_pred = corr_o_bc_pred_tensor.T.reshape((n_tau, n_configs, -1)).numpy()
        corr_o_unlab_pred = corr_o_unlab_pred_tensor.T.reshape((n_tau, n_configs, -1)).numpy()
        # corr_o_train_pred = np.swapaxes(corr_o_train_pred, 0, 1)
        # corr_o_bc_pred = np.swapaxes(corr_o_bc_pred, 0, 1)
        # corr_o_unlab_pred = np.swapaxes(corr_o_unlab_pred, 0, 1)
        corr_o_pred = []
        curr_ind_train = 0
        curr_ind_bc = 0
        curr_ind_unlab = 0
        for i in range(n_tsrc):
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
        ds_ml_ratio_method = ratio_method(
            # corr_i=np.expand_dims(corr_o_pred_uncorrected, axis=-1),
            # corr_i=corr_o_unlab_pred_tensor,
            corr_i=corr_o_pred,
            corr_o=corr_o,
            # lab_ind_list=[0],
            lab_ind_list=args.train_ind_list+args.bc_ind_list,
            modify=args.modify_ratio,
        )


    if args.compare_ratio_method == 1:
        print("gv.mean(gv_dataset['hp_o_pred']).shape:", gv.mean(ds_ratio_method['hp_o_pred']).shape)
        plt.errorbar(
            x=np.arange(0, n_tau),
            y=gv.mean(ds_ratio_method['hp_o_pred']),
            yerr=gv.sdev(ds_ratio_method['hp_o_pred']),
            c='c',
            label='Pred Ratio Method',
        )

    if args.compare_ml_ratio_method == 1:
        print("gv.mean(gv_dataset['hp_o_pred']).shape:", gv.mean(ds_ml_ratio_method['hp_o_pred']).shape)
        plt.errorbar(
            x=np.arange(0, n_tau),
            y=gv.mean(ds_ml_ratio_method['hp_o_pred']),
            yerr=gv.sdev(ds_ml_ratio_method['hp_o_pred']),
            c='m',
            label='Pred ML+Ratio Method',
        )

    plt.xlabel('Time extent')
    plt.ylabel('Correlator')
    plt.yscale('log')
    plt.legend()

    save_plot(fig=fig, path='./plots/', filename='pred_correlator')


#===================================================================================================
# RELATIVE CORRELATED DIFFERENCES 
#===================================================================================================
def relative_correlated_difference(
    num_tau: int,
    num_cfgs: int,
    dict_data: dict[str, npt.NDArray],
    dict_results: dict[str, npt.NDArray]
) -> None:
    """
    Computes and plots relative correlated differences for correlator data.
    
    Args:
        corr_o_uncorrected: Normalized, bias-uncorrected correlator averaged over source times
        corr_o_uncorrected: Normalized, bias-corrected correlator averaged over source times
    """
    n_corr_o_unlab_tensor = dict_data["n_corr_o_unlab_tensor"]
    n_corr_o_bc_tensor = dict_data["n_corr_o_bc_tensor"]
    
    n_corr_o_unlab_pred_tensor = dict_results["n_corr_o_unlab_pred_tensor"]
    n_corr_o_bc_pred_tensor = dict_results["n_corr_o_bc_pred_tensor"]

    n_corr_o_unlab_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_unlab_tensor, num_tau, num_cfgs)
    n_corr_o_unlab_pred_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_unlab_pred_tensor, num_tau, num_cfgs)

    n_corr_o_bc_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_bc_tensor, num_tau, num_cfgs)
    n_corr_o_bc_pred_vs_tau = tensor_to_avg_over_tsrc(n_corr_o_bc_pred_tensor, num_tau, num_cfgs)


    diff_n_corr_o_uncorrected = n_corr_o_unlab_pred_vs_tau - n_corr_o_unlab_vs_tau
    diff_n_corr_o_corrected = diff_n_corr_o_uncorrected + n_corr_o_bc_vs_tau - n_corr_o_bc_pred_vs_tau

    relative_diff_n_corr_o_uncorrected = np.average(diff_n_corr_o_uncorrected, axis=-1) / sem(diff_n_corr_o_uncorrected, axis=-1)
    relative_diff_n_corr_o_corrected = np.average(diff_n_corr_o_corrected, axis=-1) / sem(diff_n_corr_o_corrected, axis=-1)
    
    fig = plt.figure(figsize=(8., 6.))

    plt.plot(relative_diff_n_corr_o_uncorrected, label='Bias-uncorrected', c='b')
    plt.plot(relative_diff_n_corr_o_corrected, label='Bias-corrected', c='r')

    plt.axhline(y=1, c='k', ls='--')
    plt.axhline(y=-1, c='k', ls='--')

    plt.xlabel('Time extent')
    plt.ylabel('Rel. correlated diff.')

    plt.legend()
    save_plot(fig=fig, path='./plots/', filename='rel_correlated_diff')
    

#===================================================================================================
# NOISE TO SIGNAL RATIOS
#===================================================================================================
def noise_to_signal(*args) -> None:
    """
    DOCS TODO
    """
    fig = plt.figure(figsize=(8., 6.))

    # yscale = 'linear'
    yscale = 'log'

    plt.plot(
        np.arange(0, n_tau),
        sem(corr_o_truth, axis=-1)/np.average(corr_o_truth, axis=-1),
        c='b',
        linewidth=1.4,
        label='Truth',
    )
    plt.plot(
        np.arange(0, n_tau),
        sem(corr_o_labeled_truth, axis=-1)/np.average(corr_o_labeled_truth, axis=-1),
        c='g',
        linewidth=1.3,
        label='Labeled set',
    )
    # plt.plot(
    #     np.arange(0, n_tau),
    #     sem(corr_o_train_truth, axis=-1)/np.average(corr_o_train_truth, axis=-1),
    #     c='violet',
    #     label='Train set',
    # )
    plt.plot(
        np.arange(0, n_tau),
        sem(corr_o_pred_corrected, axis=-1)/np.average(corr_o_pred_corrected, axis=-1),
        c='r',
        linewidth=1.2,
        label='Pred Corrected',
    )
    plt.plot(
        np.arange(0, n_tau),
        sem(corr_o_pred_uncorrected, axis=-1)/np.average(corr_o_pred_uncorrected, axis=-1),
        c='gold',
        linewidth=1.1,
        label='Pred Uncorrected',
    )

    if args.compare_ratio_method == 1:
        plt.plot(
            np.arange(0, n_tau),
            gv.sdev(ds_ratio_method['hp_o_pred']) / gv.mean(ds_ratio_method['hp_o_pred']),
            c='c',
            linewidth=1.0,
            label='Pred Ratio Method',
        )

    if args.compare_ml_ratio_method == 1:
        plt.plot(
            np.arange(0, n_tau),
            gv.sdev(ds_ml_ratio_method['hp_o_pred']) / gv.mean(ds_ml_ratio_method['hp_o_pred']),
            c='m',
            linewidth=0.9,
            label='Pred ML+Ratio Method',
        )

    plt.xlabel('Time extent')
    plt.ylabel('NtS')
    plt.yscale(yscale)

    plt.legend(fontsize=12)
    save_plot(fig=fig, path='./plots/', filename='pred_nts')
    

def normalized_noise_to_signal(*args) -> None:
    """
    Docs TODO
    """
    ##### Normalized NtS #####
    fig = plt.figure(figsize=(8., 6.))
    normalization_denominator = sem(corr_o_truth, axis=-1)/np.average(corr_o_truth, axis=-1)
    plt.plot(
        np.arange(0, n_tau),
        sem(corr_o_truth, axis=-1)/np.average(corr_o_truth, axis=-1)/normalization_denominator,
        c='b',
        linewidth=1.4,
        label='Truth',
    )

    plt.plot(
        np.arange(0, n_tau),
        sem(corr_o_labeled_truth, axis=-1)/np.average(corr_o_labeled_truth, axis=-1)/normalization_denominator,
        c='g',
        linewidth=1.3,
        label='Labeled set',
    )

    plt.plot(
        np.arange(0, n_tau),
        sem(corr_o_pred_corrected, axis=-1)/np.average(corr_o_pred_corrected, axis=-1)/normalization_denominator,
        c='r',
        linewidth=1.2,
        label='Pred Corrected',
    )
    plt.plot(
        np.arange(0, n_tau),
        sem(corr_o_pred_uncorrected, axis=-1)/np.average(corr_o_pred_uncorrected, axis=-1)/normalization_denominator,
        c='gold',
        linewidth=1.1,
        label='Pred Uncorrected',
    )

    if args.compare_ratio_method == 1:
        plt.plot(
            np.arange(0, n_tau),
            gv.sdev(ds_ratio_method['hp_o_pred']) / gv.mean(ds_ratio_method['hp_o_pred'])/normalization_denominator,
            c='c',
            linewidth=1.0,
            label='Pred Ratio Method',
        )

    if args.compare_ml_ratio_method == 1:
        plt.plot(
            np.arange(0, n_tau),
            gv.sdev(ds_ml_ratio_method['hp_o_pred']) / gv.mean(ds_ml_ratio_method['hp_o_pred'])/normalization_denominator,
            c='m',
            linewidth=0.9,
            label='Pred ML+Ratio Method',
        )

    plt.xlabel('Time extent')
    plt.ylabel('NtS / (Truth NtS)')
    plt.yscale('linear')

    plt.legend(fontsize=12)
    save_plot(fig=fig, path='./plots/', filename='pred_nts_normalized')
    

def plot_error_breakdown(
    pred_corrected,
    pred_uncorrected,
    bias_correction,
    fig_name='./error_breakdown',
    truth=None,
) -> None:

    sem_pred_corrected = sem(pred_corrected, axis=-1)
    sem_pred_uncorrected = sem(pred_uncorrected, axis=-1)
    sem_bias_correction = sem(bias_correction, axis=-1)

    n_tau = sem_pred_corrected.shape[0]

    x = np.arange(0, n_tau)
    y1 = sem_pred_uncorrected / sem_pred_corrected
    y1 = y1**2
    y2 = sem_bias_correction / sem_pred_corrected
    y2 = y2**2

    fig = plt.figure(figsize=(8., 6.))

    plt.fill_between(x, y1=y1, color='lightyellow', label='Uncorrected')
    plt.fill_between(x, y1=y1, y2=y1+y2, color='thistle', label='Bias correction')

    plt.axhline(y=1.0, color='r', linestyle = '--', label='Corrected')

    if truth is not None:
        y_sem_truth = sem(truth, axis=-1) / sem_pred_corrected
        y_sem_sq_truth = y_sem_truth**2
        plt.plot(x, y_sem_sq_truth, color='b', label='Truth')

    plt.xlabel('Time extent')
    plt.ylabel(r'Normalized Error Squared')

    plt.legend()
    save_plot(fig=fig, path='', filename=fig_name)

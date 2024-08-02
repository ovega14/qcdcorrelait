import numpy as np
import gvar as gv
from scipy.stats import sem

import numpy.typing as npt

import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../')
from utils import save_plot


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
def plot_correlators(
    num_tau: int,
    corr_o_truth: npt.NDArray,
    corr_o_train_truth: npt.NDArray,
    corr_o_pred_corrected: npt.NDArray,
    corr_o_pred_uncorrected: npt.NDArray,
    ds_ratio_method: dict[str, npt.NDArray],
    ds_ml_ratio_method: dict[str, npt.NDArray],
    results_dir,
    args
) -> None:
    """
    Plots correlator data along with predicted correlators on the same figure.

    Args:
        num_tau: Number of time extents being used in the fit
        corr_o_pred_corrected: Bias-corrected correlator predicted by the ML model
        corr_o_pred_uncorrected: Bias-uncorrected correlator predicted by the ML model
        corr_o_truth: Truth-level correlator data
        corr_o_train_truth: Subset of truth data reserved for training
        ds_ratio_method: Dictionary of correlators output by ratio method
        ds_ML_ratio_method: Dictionary of correlators output by ratio method combined with ML
    """
    fig = plt.figure(figsize=(8., 6.))

    plt.errorbar(
        x = np.arange(0, num_tau),
        y = np.average(corr_o_truth, axis=-1),
        yerr = sem(corr_o_truth, axis=-1),
        c = 'b',
        label = 'Truth',
    )

    plt.errorbar(
        x = np.arange(0, num_tau),
        y = np.average(corr_o_train_truth, axis=-1),
        yerr = sem(corr_o_train_truth, axis=-1),
        c = 'g',
        label = 'Training set',
    )

    plt.errorbar(
        x = np.arange(0, num_tau),
        y = np.average(corr_o_pred_corrected, axis=-1),
        yerr = sem(corr_o_pred_corrected, axis=-1),
        c = 'r',
        label = 'Pred Corrected',
    )

    plt.errorbar(
        x = np.arange(0, num_tau),
        y = np.average(corr_o_pred_uncorrected, axis=-1),
        yerr = sem(corr_o_pred_uncorrected, axis=-1),
        c = 'gold',
        label = 'Pred Uncorrected',
    )

    # Optionally include ratio method results
    if args.compare_ratio_method == 1:
        plt.errorbar(
            x = np.arange(0, num_tau),
            y = gv.mean(ds_ratio_method['hp_o_pred']),
            yerr = gv.sdev(ds_ratio_method['hp_o_pred']),
            c = 'c',
            label = 'Pred Ratio Method',
        )
    if args.compare_ml_ratio_method == 1:
        plt.errorbar(
            x = np.arange(0, num_tau),
            y = gv.mean(ds_ml_ratio_method['hp_o_pred']),
            yerr = gv.sdev(ds_ml_ratio_method['hp_o_pred']),
            c = 'm',
            label = 'Pred ML+Ratio Method',
        )

    plt.xlabel('Time extent')
    plt.ylabel('Correlator')
    plt.yscale('log')
    plt.legend()

    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='pred_correlator')


#===================================================================================================
# RELATIVE CORRELATED DIFFERENCES 
#===================================================================================================
def plot_relative_correlated_difference(
    n_corr_o_unlab_vs_tau, 
    n_corr_o_unlab_pred_vs_tau,
    n_corr_o_bc_vs_tau,
    n_corr_o_bc_pred_vs_tau,
    results_dir
) -> None:
    """
    Computes and plots relative correlated differences for correlator data.
    
    Args:
        num_tau: Number of time extents being used for fitting
        num_cfgs: Number of configurations in the original dataset
        dict_data: Dictionary of originally preprocessed correlator data
        dict_results: Dictionary of resultant correlator data from inference
    """
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
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='rel_correlated_diff')
    

#===================================================================================================
# NOISE TO SIGNAL RATIOS
#===================================================================================================
def plot_noise_to_signal(
    num_tau: int,
    corr_o_truth,
    corr_o_labeled_truth,
    corr_o_pred_corrected,
    corr_o_pred_uncorrected,
    ds_ratio_method,
    ds_ml_ratio_method,
    results_dir,
    args
) -> None:
    """
    DOCS TODO

    Args:
        num_tau: Number of time extents used for fitting
        corr_o_truth: Truth-level correlator data
        corr_o_labeled_truth: Subset of truth data reserved for training and bias correction
        corr_o_pred_corrected: Bias-corrected correlator predicted by the ML model
        corr_o_pred_uncorrected: Bias-uncorrected correlator predicted by the ML model
        ds_ratio_method: Dictionary of correlators output by ratio method
        ds_ML_ratio_method: Dictionary of correlators output by ratio method combined with ML
    """
    fig = plt.figure(figsize=(8., 6.))

    # yscale = 'linear'
    yscale = 'log'

    plt.plot(
        np.arange(0, num_tau),
        sem(corr_o_truth, axis=-1)/np.average(corr_o_truth, axis=-1),
        c='b',
        linewidth=1.4,
        label='Truth',
    )
    plt.plot(
        np.arange(0, num_tau),
        sem(corr_o_labeled_truth, axis=-1)/np.average(corr_o_labeled_truth, axis=-1),
        c='g',
        linewidth=1.3,
        label='Labeled set',
    )

    plt.plot(
        np.arange(0, num_tau),
        sem(corr_o_pred_corrected, axis=-1)/np.average(corr_o_pred_corrected, axis=-1),
        c='r',
        linewidth=1.2,
        label='Pred Corrected',
    )
    plt.plot(
        np.arange(0, num_tau),
        sem(corr_o_pred_uncorrected, axis=-1)/np.average(corr_o_pred_uncorrected, axis=-1),
        c='gold',
        linewidth=1.1,
        label='Pred Uncorrected',
    )

    if args.compare_ratio_method == 1:
        plt.plot(
            np.arange(0, num_tau),
            gv.sdev(ds_ratio_method['hp_o_pred']) / gv.mean(ds_ratio_method['hp_o_pred']),
            c='c',
            linewidth=1.0,
            label='Pred Ratio Method',
        )

    if args.compare_ml_ratio_method == 1:
        plt.plot(
            np.arange(0, num_tau),
            gv.sdev(ds_ml_ratio_method['hp_o_pred']) / gv.mean(ds_ml_ratio_method['hp_o_pred']),
            c='m',
            linewidth=0.9,
            label='Pred ML+Ratio Method',
        )

    plt.xlabel('Time extent')
    plt.ylabel('NtS')
    plt.yscale(yscale)

    plt.legend(fontsize=12)
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='pred_nts')
    

def plot_normalized_noise_to_signal(
    num_tau: int,
    corr_o_truth,
    corr_o_labeled_truth,
    corr_o_pred_corrected,
    corr_o_pred_uncorrected,
    ds_ratio_method,
    ds_ml_ratio_method,
    results_dir,
    args
) -> None:
    """
    Docs TODO
    """
    ##### Normalized NtS #####
    fig = plt.figure(figsize=(8., 6.))
    normalization_denominator = sem(corr_o_truth, axis=-1)/np.average(corr_o_truth, axis=-1)
    plt.plot(
        np.arange(0, num_tau),
        sem(corr_o_truth, axis=-1)/np.average(corr_o_truth, axis=-1)/normalization_denominator,
        c='b',
        linewidth=1.4,
        label='Truth',
    )

    plt.plot(
        np.arange(0, num_tau),
        sem(corr_o_labeled_truth, axis=-1)/np.average(corr_o_labeled_truth, axis=-1)/normalization_denominator,
        c='g',
        linewidth=1.3,
        label='Labeled set',
    )

    plt.plot(
        np.arange(0, num_tau),
        sem(corr_o_pred_corrected, axis=-1)/np.average(corr_o_pred_corrected, axis=-1)/normalization_denominator,
        c='r',
        linewidth=1.2,
        label='Pred Corrected',
    )
    plt.plot(
        np.arange(0, num_tau),
        sem(corr_o_pred_uncorrected, axis=-1)/np.average(corr_o_pred_uncorrected, axis=-1)/normalization_denominator,
        c='gold',
        linewidth=1.1,
        label='Pred Uncorrected',
    )

    if args.compare_ratio_method == 1:
        plt.plot(
            np.arange(0, num_tau),
            gv.sdev(ds_ratio_method['hp_o_pred']) / gv.mean(ds_ratio_method['hp_o_pred'])/normalization_denominator,
            c='c',
            linewidth=1.0,
            label='Pred Ratio Method',
        )

    if args.compare_ml_ratio_method == 1:
        plt.plot(
            np.arange(0, num_tau),
            gv.sdev(ds_ml_ratio_method['hp_o_pred']) / gv.mean(ds_ml_ratio_method['hp_o_pred'])/normalization_denominator,
            c='m',
            linewidth=0.9,
            label='Pred ML+Ratio Method',
        )

    plt.xlabel('Time extent')
    plt.ylabel('NtS / (Truth NtS)')
    plt.yscale('linear')

    plt.legend(fontsize=12)
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='pred_nts_normalized')
    

#===================================================================================================
# ERROR BREAKDOWN
#===================================================================================================
def plot_error_breakdown(
    pred_corrected,
    pred_uncorrected,
    bias_correction,
    results_dir,
    fig_name = './error_breakdown',
    truth = None,
) -> None:
    """
    Docs TODO
    """
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
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename=fig_name)


#===================================================================================================
# FIT PARAMETERS (GLOBAL COMPARISON)
#===================================================================================================
def plot_fit_params(tag, filename, dict_fits, args):
    reg_methods = list(dict_fits.keys())

    # Truth data
    fit_truth = dict_fits[reg_methods[0]]['corr_o_truth']
    a_truth = fit_truth.p[filename + ':a']
    dE_truth = fit_truth.p[filename + ':dE']

    # Plot the truth & ML fit parameters as errorbars
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    font = {'weight' : 'normal',
            'size'   : 18}
    fig, axes = plt.subplots(2, 2)#, figsize=(8, 5))

    for i in range(0, 2):  # amplitudes
        axes[0, i].errorbar(x=0, y=a_truth[i].mean, yerr=a_truth[i].sdev, linestyle='None', elinewidth=0.65, capsize=1.5, capthick=0.75, fmt='o', mfc='white', ms=1.75, markeredgewidth=0.75, label='Truth')
        for j, method in enumerate(reg_methods):
            fit = dict_fits[method][tag]
            a_pred = fit.p[filename + ':a']
            axes[0, i].errorbar(x=j+1, y=a_pred[i].mean, yerr=a_pred[i].sdev, linestyle='None', elinewidth=0.5, capsize=1.5, capthick=0.75, marker='^', ms=0.75, label=f'{method}')
        if args.compare_ratio_method:
            for j, method in enumerate(reg_methods):
                fit_rm = dict_fits[method]['hp_o_pred']
                a_rm = fit_rm.p[filename + ':a']
                axes[0, i].errorbar(x=len(reg_methods)+j+1, y=a_rm[i].mean, yerr=a_rm[i].sdev, linestyle='None', elinewidth=0.5, capsize=1.5, capthick=0.75, marker='s', ms=0.75, label=f'RM + {method}')
        axes[0, i].fill_between(np.linspace(-1, 2*len(reg_methods)+1, 20), a_truth[i].mean - a_truth[i].sdev, a_truth[i].mean + a_truth[i].sdev, alpha=0.2)
        axes[0, i].set_xlim(-1, 2*len(reg_methods)+1)
        axes[0, i].set_ylabel(f'$a_{i}$')
        axes[0, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    for i in range(0, 2):  # energies
        axes[1, i].errorbar(x=0, y=dE_truth[i].mean, yerr=dE_truth[i].sdev, linestyle='None', elinewidth=0.65, capsize=1.5, capthick=0.75, fmt='o', mfc='white', ms=1.75, markeredgewidth=0.75, label='Truth')
        for j, method in enumerate(reg_methods):
            fit = dict_fits[method][tag]
            dE_pred = fit.p[filename + ':dE']
            axes[1, i].errorbar(x=j+1, y=dE_pred[i].mean, yerr=dE_pred[i].sdev, linestyle='None', elinewidth=0.5, capsize=1.5, capthick=0.75, marker='^', ms=0.75, label=f'{method}')
        if args.compare_ratio_method:
            for j, method in enumerate(reg_methods):
                fit_rm = dict_fits[method]['hp_o_pred']
                dE_rm = fit_rm.p[filename + ':dE']
                axes[1, i].errorbar(x=len(reg_methods)+j+1, y=dE_rm[i].mean, yerr=dE_rm[i].sdev, linestyle='None', elinewidth=0.5, capsize=1.5, capthick=0.75, marker='s', ms=0.75, label=f'RM + {method}')
        axes[1, i].fill_between(np.linspace(-1, 2*len(reg_methods)+1, 20), dE_truth[i].mean - dE_truth[i].sdev, dE_truth[i].mean + dE_truth[i].sdev, alpha=0.2)
        axes[1, i].set_xlim(-1, 2*len(reg_methods)+1)
        axes[1, i].set_ylabel(f'$dE_{i}$')
        axes[1, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), frameon=False, bbox_to_anchor=(1.15, 0), loc='lower left', fontsize='x-small')

    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    #plt.tight_layout()

    save_plot(fig=fig, path=args.results_dir +'/', filename='fit_params_comparison')

"""Plotting functions for tracking quantities during model training."""
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import sys
sys.path.insert(0, '../')
from utils import save_plot


# =============================================================================
#  LOSS CURVES
# =============================================================================
def plot_loss(losses: list[int], results_dir: str) -> None:
    """
    Plots training loss of a neural network over the training steps.
    
    Args:
        losses: List of losses at each training step
        results_dir: Name of directory in which to save plots
    """
    fig = plt.figure(figsize=(8., 6.))
    
    plt.plot(losses, color='firebrick')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename='training_loss')


# =============================================================================
#  CORRELATION COEFFICIENTS
# =============================================================================
def plot_diag_correlations(
    correlations: list[npt.NDArray], 
    results_dir: str
) -> None:
    """
    Plots the diagonal elements of the correlation matrix between the predicted
    and target correlator during training. 

    Args:
        correlations: List of correlation matrices at each training step
        results_dir: Name of directory in which to save plots
    """
    fig = plt.figure(figsize=(8., 6.))
    
    training_steps = len(correlations)
    for tau in range(1, 6):
        plt.plot(correlations[:, tau, 191 + tau], label=rf'$\tau={tau}$')
    plt.hlines(1.0, 0, training_steps, color='black', linestyle='dashed')
    plt.ylabel(r"$\rho(O(\tau), O^{\mathrm{pred}}(\tau))$")
    
    plt.xlabel('Training Iterations')
    plt.legend()
    save_plot(fig=fig, path=f'{results_dir}/plots/', 
              filename='diag_training_correlation')
    

def plot_off_diag_correlations(
    correlations: list[npt.NDArray],
    results_dir: str
) -> None:
    """
    Plots the off-diagonal elements of the correlation matrix between the 
    predicted and target correlator during training. 

    Args:
        correlations: List of correlation matrices at each training step
        results_dir: Name of directory in which to save plots
    """
    fig = plt.figure(figsize=(8., 6.))
    for tau in range(1, 20):
        plt.plot(correlations[:, tau, 191 + 12], label=rf'$\tau={tau}$')
        plt.plot(correlations[:, 191 + tau, 191 + 12], 
                 label=rf'Truth, $\tau={tau}$', linestyle='dashed')
    plt.ylabel(r"$\rho(O(\tau'=12), O^{\mathrm{pred}}(\tau))$")
    plt.xlabel('Training Iterations')
    plt.legend()
    save_plot(fig=fig, path=f'{results_dir}/plots/', 
              filename='off_diag_training_correlation')
    

# =============================================================================
#  CORRELATION HEATMAPS
# =============================================================================
def plot_correlation_heatmaps(
    correlations: list[npt.NDArray],
    results_dir: str
) -> None:
    """
    Plots the full correlation coefficient matrices between target and 
    predicted correlator during training as 2D heatmaps.

    Args:
        correlations: List of correlation matrices at each training step
        results_dir: Name of directory in which to save plots
    """
    fig, axes = plt.subplots(1, 5, sharey=True, figsize=(20, 4.))
    fig.supylabel(r"$\rho(O(\tau), O^{\mathrm{pred}}(\tau'))$")
    for i in range(4):
        ax = axes[i]
        im = ax.imshow(correlations[50*i], cmap='hot')
        im.norm.autoscale([0, 1])
        ax.set_xlabel(f'Iter {50*i}')
    im = axes[-1].imshow(correlations[-1], cmap='hot')
    im.norm.autoscale([0, 1])
    axes[-1].set_xlabel(f'Iter {len(correlations)}')
    #cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.])
    #fig.colorbar(im, cax=cbar_ax)  # TODO
    save_plot(fig=fig, path=f'{results_dir}/plots/', 
              filename='correlation_heatmaps')

        
def plot_final_correlation_heatmap(
    correlations: list[npt.NDArray],
    results_dir: str
) -> None:
    """Plots the final heatmap of correlation matrix after training model."""
    fig = plt.figure(figsize=(8., 8.))
    plt.title(r"$\rho(O(\tau), O^{\mathrm{pred}}(\tau'))$")
    plt.imshow(correlations[-1], cmap='hot')
    save_plot(fig=fig, path=f'{results_dir}/plots/', 
              filename='final_correlation')
    

def plot_final_diag_correlations(
    correlations: list[npt.NDArray],
    results_dir: str
) -> None:
    """
    Plots the diagonal elements of the correlation matrix between predicted
    and target correlator as a function of the time extent.

    Args:
        correlations: List of correlation matrices at each training step
        results_dir: Name of directory in which to save plots
    """
    final_corr = correlations[-1]
    num_tau = final_corr.shape[0] / 2
    taus = list(range(num_tau))
    diag_corrs = np.diag(final_corr[:num_tau, num_tau:])  # upper-right quad
    
    fig = plt.figure(figsize=(8., 6.))
    plt.plot(taus, diag_corrs)
    plt.hlines(1.0, 0, num_tau, color='black', linestyle='dashed')
    plt.xlabel(r'Time extent, $\tau$')
    plt.ylabel(r'$\rho(O(\tau), O^{\rm pred}(\tau))$')
    save_plot(fig=fig, path=f'{results_dir}/plots/', 
              filename='correlation_vs_tau_after_training')
    
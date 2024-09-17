"""Plotting functions for tracking quantities during model training."""
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional
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
def plot_correlations(
    correlations: list[npt.NDArray], 
    results_dir: str,
    *,
    tau_1: int,
    tau_2: Optional[int] = None
) -> None:
    """
    Plots the element of the correlation matrix specified by the pair 
    `(tau_1, tau_2)` between the predicted and target correlator during 
    training. If no value of `tau_2` is provided, then defaults to the diagonal
    element `(tau_1, tau_1)`. 

    Args:
        correlations: List of correlation matrices at each training step
        results_dir: Name of directory in which to save plots
        tau_1: First lattice time at which to observe correlations
        tau_2: Second lattice time at which to observe correlations
    """
    fig = plt.figure(figsize=(8., 6.))
    
    num_tau = int(correlations[0].shape[1] / 2)
    if tau_2 is None:
        tau_2 = tau_1

    plt.plot(correlations[:, tau_1, num_tau - 1 + tau_2], ls='dashed')
    plt.plot(correlations[:, num_tau - 1 + tau_1, num_tau - 1 + tau_2], 
             ls='dashed')
    #plt.hlines(1.0, 0, len(correlations), color='black', linestyle='dashed')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Training Iterations')
    title = r"$\rho(O(\tau={{{tau_1}}}), O^{{\rm pred}}(\tau'={{{tau_2}}}))$"
    plt.title(title.format(tau_1=tau_1, tau_2=tau_2))
    
    if tau_2 != tau_1:
        filename = 'off_diag_training_correlation'
    else: 
        filename = 'diag_training_correlation'
    save_plot(fig=fig, path=f'{results_dir}/plots/', filename=filename)
    

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

    # Get just the upper right quadrant
    num_tau = int(correlations[-1].shape[0] / 2)
    final_corrs = correlations[-1][:num_tau, num_tau:]
    plt.imshow(final_corrs, cmap='viridis')
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
    num_tau = int(final_corr.shape[0] / 2)
    taus = list(range(num_tau))
    diag_corrs = np.diag(final_corr[:num_tau, num_tau:])  # upper-right quad
    
    fig = plt.figure(figsize=(8., 6.))
    plt.plot(taus, diag_corrs)
    plt.hlines(1.0, 0, num_tau, color='black', linestyle='dashed')
    plt.xlabel(r'Time extent, $\tau$')
    plt.ylabel(r'$\rho(O(\tau), O^{\rm pred}(\tau))$')
    save_plot(fig=fig, path=f'{results_dir}/plots/', 
              filename='correlation_vs_tau_after_training')
    
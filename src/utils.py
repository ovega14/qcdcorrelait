import numpy as np
import matplotlib.pyplot as plt
import pickle

import numpy.typing as npt
from typing import TypeVar

Regressor = TypeVar('Regressor')


#==============================================================================
# SEEDING
#==============================================================================
def set_np_seed(seed: int) -> None:
    """
    Sets the `numpy` random seed.

    Args:
        seed: The random seed
    """
    print("numpy seed is set to {}".format(seed))
    np.random.seed(seed)


#==============================================================================
# SAVE DATA
#==============================================================================
def save_model(
    model: Regressor,
    path: str
) -> None:
    """
    Saves the learned parameters of a trained regression model.
    
    Args: 
        model: A trained Regressor object, either from PyTorch or Sklearn
    """
    pickle.dump(
        model, 
        open(f'{path}/model.pkl', 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL
    )


def save_results(
    dict_results: dict[str, npt.NDArray], 
    path: str
) -> None:
    """
    Saves the results of the fit on correlator data.
    
    Args:
        dict_results: Dictionary of resulting data from correlator prediction
        path: Path to directory in which to save results
    """
    pickle.dump(
        dict_results, 
        open(path + '.pkl', 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL
    )


def save_plot(
    fig,
    path='./',
    filename=None,
    save_as_pdf=True,
    save_as_png=True,
    save_as_svg=True,
    dpi=800.,
    tight_layout=True,
    show=False,
) -> None:
    """
    Args:
        fig: matplotlib figure object
        path: The path where the figure is saved, ending with ``/``.
        filename: The file name, not including the file type suffix, ending 
            with no ``.``.
    """
    if tight_layout:
        plt.tight_layout()
    if save_as_pdf:
        fig.savefig(
            path+filename+'.pdf',
            dpi=dpi,
        )
    if save_as_png:
        fig.savefig(
            path+filename+'.png',
            dpi=dpi,
        )
    if save_as_svg:
        fig.savefig(
            path+filename+'.svg',
        )
    if show:
        plt.show()
    plt.close()

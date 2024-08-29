import numpy as np
import matplotlib.pyplot as plt
import pickle

import numpy.typing as npt
from typing import TypeVar

Regressor = TypeVar('Regressor')


# =============================================================================
#  SEEDING
# =============================================================================
def set_np_seed(seed: int) -> None:
    """
    Sets the `numpy` random seed.

    Args:
        seed: The random seed
    """
    print("numpy seed is set to {}".format(seed))
    np.random.seed(seed)


# =============================================================================
#  MODEL I/O UTILS
# =============================================================================
def save_model(
    model: Regressor,
    path: str
) -> None:
    """
    Saves the learned parameters of a trained regression model.
    
    Args: 
        model: A trained Regressor object, either from PyTorch or Sklearn
        path: Name of the directory in which to store the model
    """
    pickle.dump(
        model, 
        open(f'{path}/model.pkl', 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL
    )


def load_model(
    path: str
) -> Regressor:
    """
    Loads a trained model back into memory for usage, i.e. inference. Assumes
    that the model is saved in a file 'path/model.pkl'. 

    Args:
        path: Name of the directory in which the model is stored

    Returns:
        A trained regressor
    """
    with open(f'{path}/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


# =============================================================================
#  DATASET I/O UTILS
# =============================================================================
def save_data(
    data_dict: dict[str, npt.NDArray], 
    path: str
) -> None:
    """
    Saves a dictionary of correlator data.
    
    Args:
        data_dict: Dictionary of correlator data
        path: Path to directory in which to save data
    """
    pickle.dump(
        data_dict, 
        open(path + '.pkl', 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL
    )


def load_data(path: str) -> dict[str, npt.NDArray]:
    """
    Loads into memory a dictionary of correlator data.

    Args:
        path: Name of the directory where the data is stored

    Returns:
        A dictionary of correlator data
    """
    with open(path + '.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


# =============================================================================
#  SAVING PLOTS
# =============================================================================
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

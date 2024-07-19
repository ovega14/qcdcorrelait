import gvar as gv
import numpy as np
import numpy.typing as npt
from typing import List, Optional, TypeVar
GVDataset = TypeVar('GVDataset')
import copy

import sys
sys.path.insert(0, '../')
from processing.preprocessing import convert_to_gvars


class RatioMethod:
    """
    Unboosted ratio method.

    Args:
        TODO
    
    Attributes:
        TODO
    """
    def __init__(
        self, 
        corr_2pt_i: npt.NDArray, 
        corr_2pt_o: npt.NDArray, 
        lab_ind_list: List[int]
    ):
        lp_i = corr_2pt_i[:, :, lab_ind_list]
        lp_o = corr_2pt_o[:, :, lab_ind_list]
        corrs: list[npt.NDArray] = [corr_2pt_i, corr_2pt_o, lp_i, lp_o]
        
        for ix, corr in enumerate(corrs):
            corrs[ix] = np.average(corr, axis=-1)

        dict_orig_corrs: dict[str, npt.NDArray] = dict()
        names = ['hp_i', 'hp_o', 'lp_i', 'lp_o']
        for name, corr in zip(names, corrs):
            dict_orig_corrs[name] = corr
        
        self.gv_dataset = convert_to_gvars(dict_orig_corrs, averages_tsrc=False)

    @staticmethod
    def __predict(
        X_lp: npt.NDArray, 
        X_hp: npt.NDArray, 
        Y_lp: npt.NDArray
    ) -> npt.NDArray:
        return Y_lp * X_hp / X_lp
    
    def predict(self):
        X_lp = self.gv_dataset['lp_i']
        X_hp = self.gv_dataset['hp_i']
        Y_lp = self.gv_dataset['lp_o']

        Y_hp = self.__predict(X_lp, X_hp, Y_lp)
        self.gv_dataset['hp_o_pred'] =  Y_hp
        return self.gv_dataset


def ratio_method(
    corr_2pt_i: npt.NDArray, 
    corr_2pt_o: npt.NDArray, 
    lab_ind_list: List[int]
) -> GVDataset:
    """Unboosted ratio method"""
    lp_i = corr_2pt_i[:, :, lab_ind_list]
    lp_o = corr_2pt_o[:, :, lab_ind_list]
    corrs: list[npt.NDArray] = [corr_2pt_i, corr_2pt_o, lp_i, lp_o]
    
    for ix, corr in enumerate(corrs):
        corrs[ix] = np.average(corr, axis=-1)

    dict_orig_corrs: dict[str, npt.NDArray] = dict()
    names = ['hp_i', 'hp_o', 'lp_i', 'lp_o']
    for name, corr in zip(names, corrs):
        dict_orig_corrs[name] = corr
    gv_dataset = convert_to_gvars(dict_orig_corrs, averages_tsrc=False)
    
    gv_dataset["hp_o_pred"] = (gv_dataset["lp_o"]/gv_dataset["lp_i"])*gv_dataset["hp_i"]
    return gv_dataset


def boosted_ratio_method(
    corr_i: npt.NDArray, 
    corr_o: npt.NDArray, 
    lab_ind_list: List[int],
    alpha: Optional[float] = None,
    numerical_truncation: Optional[float] = 1e-50
) -> GVDataset:
    """Boosted ratio method."""
    lp_i = corr_i[:, :, lab_ind_list]
    lp_o = corr_o[:, :, lab_ind_list]
    corrs: list[npt.NDArray] = [corr_i, corr_o, lp_i, lp_o]
    
    for ix, corr in enumerate(corrs):
        corrs[ix] = np.average(corr, axis=-1)

    dict_orig_corrs: dict[str, npt.NDArray] = dict()
    names = ['hp_i', 'hp_o', 'lp_i', 'lp_o']
    for name, corr in zip(names, corrs):
        dict_orig_corrs[name] = corr
    gv_dataset = convert_to_gvars(dict_orig_corrs, averages_tsrc=False)

    num_tsrc = corr_i.shape[-1]
    unlab_inds = set(list(range(num_tsrc))) - set(lab_ind_list)
    unlab_ind_list: List[int] = list(unlab_inds)

    stn_hp_i = gv.sdev(gv_dataset['hp_i']) / gv.mean(gv_dataset['hp_i'])
    stn_lp_i = gv.sdev(gv_dataset['lp_i']) / gv.mean(gv_dataset['lp_i'])

    s = stn_hp_i / stn_lp_i

    x = gv.sdev(gv_dataset['lp_i']) / gv.mean(gv_dataset['lp_i'])
    y = gv.sdev(gv_dataset['lp_o']) / gv.mean(gv_dataset['lp_o'])

    rho = gv.evalcorr(gv_dataset)[('lp_i', 'lp_o')]
    rho = np.array([rho[i, i] for i in range(rho.shape[0])])

    if alpha is None:
        alpha = y / x / (1. + s**2)
    
    gv_hp_i_alpha = gv_dataset['hp_i'] ** alpha
    alpha_truncated = copy.deepcopy(alpha)

    for j in range(gv_hp_i_alpha.shape[0]):
        gv_hp_i_alpha_j_mean = gv.mean(gv_hp_i_alpha[j])
        gv_hp_i_alpha_j_std = gv.sdev(gv_hp_i_alpha[j])
        
        five_sigma_lower = gv_hp_i_alpha_j_mean - 5 * gv_hp_i_alpha_j_std
        five_sigma_higher = gv_hp_i_alpha_j_mean + 5 * gv_hp_i_alpha_j_std
        is_inside_five_sigma: bool = (five_sigma_lower < 0 and five_sigma_higher > 0)
        is_truncated: bool = (np.abs(gv_hp_i_alpha_j_mean) < numerical_truncation)

        if is_inside_five_sigma or is_truncated or x[j] < 0 or y[j] < 0:
            alpha_truncated[j] = 1.
    
    ratio_unmod = gv_dataset['hp_i'] / gv_dataset['lp_i']
    ratio_mod = gv_dataset['hp_i']**alpha_truncated / gv_dataset['lp_i']**alpha_truncated
    ratio = np.where(alpha_truncated != None, ratio_mod, ratio_unmod)
    
    gv_dataset['hp_o_pred'] = gv_dataset['lp_o'] * ratio
    return gv_dataset
        
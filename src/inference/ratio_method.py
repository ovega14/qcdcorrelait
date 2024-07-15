import numpy as np
import numpy.typing as npt
from typing import List

from ..processing.preprocessing import convert_to_gvars


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
        
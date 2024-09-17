import gvar as gv
import numpy as np
import numpy.typing as npt
from typing import List, Optional, TypeVar
GVDataset = TypeVar('GVDataset')
import copy

import sys
sys.path.insert(0, '../')
from processing.conversion import convert_to_gvars


class RatioMethod:
    """
    Ratio Estimator for inferring new correlator data.

    Args:
        corr_i: Input correlator data
        corr_o: Target (labeled) correlator data
        lab_ind_list: List of source time indices corresponding to labeled data
        use_ml: Whether to augment the ratio estimator with ML predictions
        boosted: Whether to use the 'boosted' ratio method
        alpha: Boost parameter (exponent)

    Attributes:
        gv_dataset: `gvar` dataset object containing correlator data by tags
        boosted: As above.
        alpha: As above.
        output_tag: Name used for inferred data. If `use_ml` is true, then the
            name indicates ML was used. Otherwise, default tag is used.
    """
    def __init__(
        self,
        corr_i: npt.NDArray,
        corr_o: npt.NDArray,
        lab_ind_list: List[int],
        use_ml: bool,
        boosted: bool,
        alpha: Optional[npt.NDArray] = None
    ):
        lp_i = corr_i[:, :, lab_ind_list]
        lp_o = corr_o[:, :, lab_ind_list]
        corrs: list[npt.NDArray] = [corr_i, corr_o, lp_i, lp_o]
        
        for ix, corr in enumerate(corrs):
            corrs[ix] = np.average(corr, axis=-1)

        dict_orig_corrs: dict[str, npt.NDArray] = dict()
        names = ['hp_i', 'hp_o', 'lp_i', 'lp_o']
        for name, corr in zip(names, corrs):
            dict_orig_corrs[name] = corr
        
        self.gv_dataset = convert_to_gvars(dict_orig_corrs, averages_tsrc=False)

        self.boosted = boosted
        self.alpha = alpha

        self.output_tag: str = 'ratio_method_pred'
        if use_ml:
            self.output_tag = 'ml_' + self.output_tag

    def _truncate_alpha(
        self, 
        numerical_truncation: Optional[float] = 1e-50
    ) -> None:
        """
        Numerically truncates the boost power `alpha`.
        """
        # Noise-to-Signal ratios
        nts_hp_i = gv.sdev(self.gv_dataset['hp_i']) / gv.mean(self.gv_dataset['hp_i'])
        nts_lp_i = gv.sdev(self.gv_dataset['lp_i']) / gv.mean(self.gv_dataset['lp_i'])
        nts_lp_o = gv.sdev(self.gv_dataset['lp_o']) / gv.mean(self.gv_dataset['lp_o'])
        
        s = nts_hp_i / nts_lp_i
        x = nts_lp_i
        y = nts_lp_o

        if self.alpha is None:
            self.alpha = y / x / (1. + s**2)
        gv_hp_i_alpha = self.gv_dataset['hp_i'] ** self.alpha
        alpha_truncated = copy.deepcopy(self.alpha)

        for j in range(gv_hp_i_alpha.shape[0]):
            gv_hp_i_alpha_j_mean = gv.mean(gv_hp_i_alpha[j])
            gv_hp_i_alpha_j_std = gv.sdev(gv_hp_i_alpha[j])
            
            five_sigma_lower = gv_hp_i_alpha_j_mean - 5 * gv_hp_i_alpha_j_std
            five_sigma_higher = gv_hp_i_alpha_j_mean + 5 * gv_hp_i_alpha_j_std
            is_inside_five_sigma: bool = (five_sigma_lower < 0 and five_sigma_higher > 0)
            is_truncated: bool = (np.abs(gv_hp_i_alpha_j_mean) < numerical_truncation)

            if is_inside_five_sigma or is_truncated or x[j] < 0 or y[j] < 0:
                alpha_truncated[j] = 1.
        self.alpha = alpha_truncated
    
    def fit(self) -> npt.NDArray:
        """
        Constructs the ratio from given data.
        """
        return self.gv_dataset['hp_i'] / self.gv_dataset['lp_i']
        
    def predict(self):
        """
        Computes the predicted output correlator using the ratio estimator.
        """
        self._truncate_alpha()
        ratio = self.fit()
        self.gv_dataset[self.output_tag] = self.gv_dataset['lp_o'] * ratio
        if self.boosted:
            boosted_ratio = (self.gv_dataset['hp_i'] / self.gv_dataset['lp_i']) ** self.alpha
            boosted_ratio = np.where(self.alpha != None, boosted_ratio, ratio)
            self.gv_dataset[self.output_tag + '_modified'] = self.gv_dataset['lp_o'] * boosted_ratio
        return self.gv_dataset
        
"""
Script to compare results generated from a family of models. 
NOTE: Assumes all models have already been trained, correlators fit, and 
individual results saved.
"""
#!/usr/bin/env python3
import argparse
import sys
sys.path.insert(0, '../src/')
from analysis.tabulation import FitParamsTable
from analysis.plotting import plot_fit_params
from utils import set_plot_preferences, load_data


# =============================================================================
TAGS: list[str] = [
    'corr_o_pred_corrected',
    'corr_o_pred_uncorrected',
]


if __name__ == '__main__':
    set_plot_preferences()


# =============================================================================
def main(args):
    global TAGS

    filename = args.input_dataname + '_' + args.output_dataname
    reg_fits: dict = {}
    for reg_method in args.reg_methods:
        path = args.results_dir + '/' + reg_method + '/all_dict_fits'
        reg_fits[reg_method] = load_data(path)

    for tag in TAGS:
        fits_table = FitParamsTable(reg_fits, args.reg_methods, tag, filename)
        with open(args.results_dir + '/fit_params_table.txt', 'a') as f:
            print(tag, file=f)
            print(fits_table, file=f)
        plot_fit_params(tag, filename, reg_fits, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--reg_methods', nargs='+', type=str)
    add('--input_dataname', type=str)
    add('--output_dataname', type=str)
    add('--results_dir', type=str)
    add('--compare_ratio_method', type=int, default=1)
    add('--compare_ml_ratio_method', type=int, default=1)

    args = parser.parse_args()
    main(args)

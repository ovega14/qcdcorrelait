import torch
import numpy as np

import argparse
import json
import copy

import sys
sys.path.insert(0, '../src/')
from utils import save_model, save_results, set_np_seed
from processing.preprocessing import get_corrs, preprocess_data
from inference.train import train_model
from inference.inference import predict
from analysis.analyze import analysis_pred

def main(args):
    ### INITIALIZATION -----------------------------------------------------------------------------
    seed = args.seed
    set_np_seed(seed)
    torch.set_default_dtype(torch.float64)  # essential for high-precision, small-value correlators

    use_torch = args.use_torch
    hdf5_filename = args.hdf5_filename
    input_dataname = args.input_dataname
    output_dataname = args.output_dataname
    train_ind_list = args.train_ind_list
    bc_ind_list = args.bc_ind_list
    reg_method = args.reg_method
    rel_eps = args.rel_eps
    dict_hyperparams = json.loads(args.dict_hyperparams)

    corr_i, corr_o = get_corrs(
        args.hdf5_filename,
        *[args.input_dataname, args.output_dataname]
    )

    num_tau, num_cfgs, num_tsrc = corr_i.shape
    print('num_tau =', num_tau)
    print('num_cfgs =', num_cfgs)
    print('num_tsrc =', num_tsrc)

    train_ind_list = args.train_ind_list
    bc_ind_list = args.bc_ind_list
    labeled_ind_list = np.sort(train_ind_list + bc_ind_list).tolist()

    unlab_ind_list = []

    for i in range(num_tsrc):
        if i not in labeled_ind_list:
            unlab_ind_list.append(i)

    dict_data = preprocess_data(
        corr_i, corr_o, train_ind_list, bc_ind_list, unlab_ind_list,
    )

    model = train_model(
        dict_data,
        args,
    )

    save_model(model=model)

    n_corr_i_train_tensor = (dict_data["corr_i_train_tensor"] - dict_data["corr_i_train_means"]) / dict_data["corr_i_train_stds"]
    dict_results = predict(
        n_corr_i_train_tensor=n_corr_i_train_tensor,
        model=model,
        args=args,
        dict_data=dict_data,
    )

    if args.save_results == 1:
        save_results(dict_results=dict_results)

    dict_data["n_corr_o_train_tensor"] = (dict_data["corr_o_train_tensor"] - dict_data["corr_o_train_means"]) / dict_data["corr_o_train_stds"]

    dict_data["n_corr_i_unlab_tensor"] = (dict_data["corr_i_unlab_tensor"] - dict_data["corr_i_train_means"]) / dict_data["corr_i_train_stds"]
    dict_data["n_corr_o_unlab_tensor"] = (dict_data["corr_o_unlab_tensor"] - dict_data["corr_o_train_means"]) / dict_data["corr_o_train_stds"]

    dict_data["n_corr_i_bc_tensor"] = (dict_data["corr_i_bc_tensor"] -  dict_data["corr_i_train_means"]) /  dict_data["corr_i_train_stds"]
    dict_data["n_corr_o_bc_tensor"] = (dict_data["corr_o_bc_tensor"] -  dict_data["corr_o_train_means"]) /  dict_data["corr_o_train_stds"]

    analysis_pred(
        corr_i, corr_o,
        num_tau, num_cfgs, num_tsrc,
        args,
        dict_data,
        dict_results,
    )
# --------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--seed', type=int, default=42)
    add('--compare_ratio_method', type=int, default=1)
    add('--compare_ml_ratio_method', type=int, default=1)
    add('--use_torch', type=int, default=1)
    add('--save_results', type=int, default=0)
    add('--hdf5_filename', type=str, default='../../l64192a_run2_810-6996_1028cfgs.hdf5')
    add('--input_dataname', type=str, default='P5-P5_RW_RW_d_d_m0.164_m0.01555_p000')
    add('--output_dataname', type=str, default='P5-P5_RW_RW_d_d_m0.164_m0.00311_p000')
    add('--train_ind_list', type=str, default='[0]')
    add('--bc_ind_list', type=str, default='[3, 6, 12, 15, 18]')
    add('--reg_method', type=str, default='MLP')
    add('--dict_hyperparams', type=str, default='{"lr": 0.01, "l2_coeff": 1e-2, "training_steps": 500}')
    add('--rel_eps', type=float, default=1e-2)
    add('--modify_ratio', type=int, default=1)

    args = parser.parse_args()

    with open('./data/commandline_args.dat', 'w') as f:
        args_dict = copy.deepcopy(args.__dict__)
        args_dict['dict_hyperparams'] = json.loads(args.dict_hyperparams)
        json.dump(args_dict, f, indent=2)

    args = vars(args)
    for key in ['train_ind_list', 'bc_ind_list']:
        if key in args.keys():
            args[key] = eval(args[key])
    args = argparse.Namespace(**args)
    main(args)

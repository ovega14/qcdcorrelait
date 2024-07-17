def analysis_pred(
        corr_i, corr_o,
        n_tau, n_configs, n_tsrc,
        args,
        dict_data,
        dict_results,
    ) -> None:
    plot_error_breakdown(
        pred_corrected=corr_o_pred_corrected,
        pred_uncorrected=corr_o_pred_uncorrected,
        bias_correction=corr_o_bias_correction,
        fig_name='./plots/error_breakdown',
        truth=corr_o_truth,
    )

    dict_orig_corrs = dict()
    print("corr_o_truth.shape", corr_o_truth.shape)
    corrs = [corr_o_truth, corr_o_pred_corrected, corr_o_pred_uncorrected]
    # for (ix, corr) in enumerate(corrs):
    #     corrs[ix] = np.average(corr, axis=-1)
    names = ["corr_o_truth", "corr_o_pred_corrected", "corr_o_pred_uncorrected"]
    for name, corr in zip(names, corrs):
        dict_orig_corrs[name] = corr

    dict_opts = dict()
    opt_keys = ['l', 'tp', 'tmin', 'tmax', 'ne', 'no', 'maxit', 'averages_tsrc']
    opt_vals = [args.input_dataname + '_' + args.output_dataname, n_tau, 2, n_tau-2, 5, 5, 5_000, False]
    for key, val in zip(opt_keys, opt_vals):
        dict_opts[key] = val
    l = dict_opts.get('l')
    dict_fits = fit_corrs(dict_orig_corrs, dict_opts)

    # Save fits for later aggregate analysis
    total_dict_fits = dict_fits

    # Write results
    with open('./results/fits.txt', 'w') as f:
        for tag in dict_fits.keys():
            print(tag, file=f)
            print(dict_fits[tag], file=f)
    with open('./results/latex_table.txt', 'w') as f:
        for tag in dict_fits.keys():
            fit = dict_fits[tag]
            print(tag + ':\n', file=f)

            a = fit.p[l + ':a']
            dE = fit.p[l + ':dE']
            chi2_dof = fit.chi2 / fit.dof
            Q = fit.Q

            string = 'Reg Method & '
            for i in range(2):
                string += f'{a[i]} & {dE[i]} & '
            string += f'{round(chi2_dof, 2)} & {round(Q, 2)} \\\\'
            print(string, file=f)
            print('=' * 120, file=f)

    # print("ds_ratio_method =", ds_ratio_method)
    if args.compare_ratio_method == 1:
        dict_fits = fit_corrs(
            dict_corrs=None,
            dict_opts=dict_opts,
            gv_ds=ds_ratio_method,
            excluding_tags=['hp_i', 'lp_i']
        )
        with open('./results/fits.txt', 'a') as f:
            for tag in dict_fits.keys():
                if tag == 'hp_o_pred':
                    print("ratio_method_pred", file=f)
                elif tag == 'hp_o_pred_modified':
                    print("ratio_method_pred_modified", file=f)
                else:
                    print(tag, file=f)
                print(dict_fits[tag], file=f)
        with open('./results/latex_table.txt', 'a') as f:
            for tag in dict_fits.keys():
                fit = dict_fits[tag]
                print(tag + ':\n', file=f)

                a = fit.p[l + ':a']
                dE = fit.p[l + ':dE']
                chi2_dof = fit.chi2 / fit.dof
                Q = fit.Q

                string = 'Reg Method & '
                for i in range(2):
                    string += f'{a[i]} & {dE[i]} & '
                string += f'{round(chi2_dof, 2)} & {round(Q, 2)}' +' \\\\'
                print(string, file=f)
                print('=' * 120, file=f)
        # Add dict entry for pure ratio method
        total_dict_fits['ratio_method_pred'] = dict_fits['hp_o_pred']
        total_dict_fits['ratio_method_pred_modified'] = dict_fits['hp_o_pred_modified']

    if args.compare_ml_ratio_method == 1:
        dict_fits = fit_corrs(
            dict_corrs=None,
            dict_opts=dict_opts,
            gv_ds=ds_ml_ratio_method,
            excluding_tags=['hp_o', 'hp_i', 'lp_i', 'lp_o']
        )
        with open('./results/fits.txt', 'a') as f:
            for tag in dict_fits.keys():
                if tag == 'hp_o_pred':
                    print("ml_ratio_method_pred", file=f)
                elif tag == 'hp_o_pred_modified':
                    print("ml_ratio_method_pred_modified", file=f)
                else:
                    print(tag, file=f)
                print(dict_fits[tag], file=f)
        with open('./results/latex_table.txt', 'a') as f:
            for tag in dict_fits.keys():
                fit = dict_fits[tag]
                print(tag + ':\n', file=f)

                a = fit.p[l + ':a']
                dE = fit.p[l + ':dE']
                chi2_dof = fit.chi2 / fit.dof
                Q = fit.Q

                string = 'Reg Method & '
                for i in range(2):
                    string += f'{a[i]} & {dE[i]} & '
                string += f'{round(chi2_dof, 2)} & {round(Q, 2)}' +' \\\\'
                print(string, file=f)
                print('=' * 120, file=f)
        # Add dict entry for ratio method + ML
        #total_dict_fits.update(dict_fits)
        total_dict_fits['ml_ratio_method_pred'] = dict_fits['hp_o_pred']
        total_dict_fits['ml_ratio_method_pred_modified'] = dict_fits['hp_o_pred_modified']
    output_filename = f'../../aggregate_results/{args.reg_method}.p'
    gv.dump(total_dict_fits, output_filename)

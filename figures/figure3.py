import itertools

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest

from figures.util import select_strain, experiments, extract_data, mcc, get_subplots, produce_figure, \
    plot_grid, plot_annotations, plot_scatter


def generate_qval_data(strain_data):
    regions = list(strain_data[list(strain_data.keys())[0]].keys())
    relevant_exps = experiments[experiments.id.isin(strain_data.keys())]
    males = relevant_exps[relevant_exps.gender == 'M']
    females = relevant_exps[relevant_exps.gender == 'F']
    male_ids = [str(i) for i in males.id.tolist()]
    female_ids = [str(i) for i in females.id.tolist()]
    qvals = pd.DataFrame()
    qvals.loc[:, 'region'] = regions
    qvals.loc[:, 'region_name'] = [r['name'] for r in mcc.get_structure_tree().get_structures_by_acronym(regions)]

    for param in ['count3d', 'density3d', 'coverage', "volume"]:
        male_data = {r: [extract_data(strain_data[e][r], param, 'median') for e in male_ids] for r in regions}
        female_data = {r: [extract_data(strain_data[e][r], param, 'median') for e in female_ids] for r in
                       regions}
        qvals.loc[:, f'{param}_male_means'] = [np.mean(male_data[r]) for r in regions]
        qvals.loc[:, f'{param}_female_means'] = [np.mean(female_data[r]) for r in regions]
        for method_name, method in {'t-test': scipy.stats.ttest_ind, 'ranksum': scipy.stats.ranksums}.items():
            pvals = [method(male_data[r], female_data[r]).pvalue for r in regions]
            rejected, pval_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.1)
            signs = np.sign(np.array(qvals[f'{param}_male_means']) - np.array(qvals[f'{param}_female_means']))
            qvals.loc[:, f"male_vs_female_{param}_{method_name}_rejected"] = rejected
            qvals.loc[:, f"male_vs_female_{param}_{method_name}_qval"] = pval_corrected * signs
            qvals.loc[:, f"male_vs_female_{param}_{method_name}_log10(qval)"] = -np.log10(pval_corrected) * signs

    for gender, gender_ids in {'male': male_ids, 'female': female_ids}.items():
        for param in ['count', 'density', 'region_area']:
            left_data = {r: [extract_data(strain_data[e][r], f"{param}_left", 'mean') for e in gender_ids] for r in
                         regions}
            right_data = {r: [extract_data(strain_data[e][r], f"{param}_right", 'mean') for e in gender_ids] for r in
                          regions}
            qvals.loc[:, f'{param}_{gender}_left_means'] = [np.mean(left_data[r]) for r in regions]
            qvals.loc[:, f'{param}_{gender}_right_means'] = [np.mean(right_data[r]) for r in regions]
            for method_name, method in {'t-test': scipy.stats.ttest_ind, 'ranksum': scipy.stats.ranksums}.items():
                pvals = [method(left_data[r], right_data[r]).pvalue for r in regions]
                rejected, pval_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.1)
                signs = np.sign(
                    np.array(qvals[f'{param}_{gender}_left_means']) - np.array(qvals[f'{param}_{gender}_right_means']))
                qvals.loc[:, f"left_{param}_{gender}_rejected"] = rejected
                qvals.loc[:, f"left_vs_right_{param}_{gender}_{method_name}_rejected"] = rejected
                qvals.loc[:, f"left_vs_right_{param}_{gender}_{method_name}_qval"] = pval_corrected * signs
                qvals.loc[:, f"left_vs_right_{param}_{gender}_{method_name}_log10(qval)"] = -np.log10(
                    pval_corrected) * signs

    return female_ids, male_ids, qvals


def plot_qval_vs_qval(qvals, prefix, highlight_col, x_col, y_col, filename):
    regions = qvals.region.tolist()
    fig, ax = get_subplots()
    highlight = qvals[prefix + "_" + highlight_col + '_qval'].to_numpy().copy()
    highlight_rejected = qvals[prefix + "_" + highlight_col + '_rejected'].to_numpy()
    highlight[~highlight_rejected] = 1
    x = qvals[prefix + "_" + x_col + '_log10(qval)'].to_numpy()
    y = qvals[prefix + "_" + y_col + '_log10(qval)'].to_numpy()
    highlight = (np.abs(highlight) <= 0.01)
    plot_scatter(ax, x[~highlight], y[~highlight], label=f'{highlight_col} not significant')
    plot_annotations(ax, np.array(regions)[highlight], x[highlight], y[highlight], dot_color='r')
    plot_grid(ax, [0], [0])
    produce_figure(ax, fig, prefix + '_' + filename)


def produce_volcano_plot(qvals, prefix, x_col1, x_col2, y_col, filename, hide_rejected=False):
    fig, ax = get_subplots()
    if hide_rejected:
        qvals = qvals[qvals[prefix + '_' + y_col + '_rejected']]
    regions = qvals.region.tolist()
    x = np.array(qvals[x_col1] - qvals[x_col2]) / np.minimum(qvals[x_col1].to_numpy(), qvals[x_col2].to_numpy())
    y = np.abs(np.array(qvals[prefix + '_' + y_col + '_log10(qval)']))
    plot_scatter(ax, x, y)
    ann = np.where((np.abs(x) > 0.05) & (y > 2))
    plot_annotations(ax, np.array(regions)[ann], x[ann], y[ann], dot_color='r')
    plot_grid(ax, horiz_lines=[2], vert_lines=[0.05, -0.05])
    produce_figure(ax, fig, prefix + '_' + filename)


def figure3(data, strain='C57BL/6J'):
    if strain is not None:
        strain_data = select_strain(data, strain)
    else:
        strain_data = data

    female_ids, male_ids, qvals = generate_qval_data(strain_data)
    params = [('count', 'density'), ('density', 'count')]
    genders = ['male', 'female']

    for param, highlight_param in params:
        produce_volcano_plot(qvals, 'male_vs_female',
                             f'{param + "3d"}_male_means',
                             f'{param + "3d"}_female_means',
                             f'{param + "3d"}_ranksum',
                             f'{param + "3d"}_qval_vs_normalized_M-F')
        plot_qval_vs_qval(qvals, 'male_vs_female',
                          f'{highlight_param + "3d"}_ranksum',
                          f'{param + "3d"}_ranksum',
                          'volume_ranksum',
                          f'{param + "3d"}_qval_vs_volume_qval')

    for (param, highlight_param), gender in itertools.product(params, genders):
        produce_volcano_plot(qvals, 'left_vs_right',
                             f'{param}_{gender}_left_means',
                             f'{param}_{gender}_right_means',
                             f'{param}_{gender}_ranksum',
                             f"{param}_{gender}_qval_vs_normalized_M-F")
        plot_qval_vs_qval(qvals, 'left_vs_right',
                          f'{highlight_param}_{gender}_ranksum',
                          f'{param}_{gender}_ranksum',
                          f'region_area_{gender}_ranksum',
                          f"{param}_{gender}_qval_vs_volume_qval")

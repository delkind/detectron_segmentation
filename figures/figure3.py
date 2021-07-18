import itertools

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest

from figures.util import experiments, mcc, get_subplots, produce_figure, plot_grid, plot_annotations, plot_scatter, \
    to_col_name


def generate_qval_data(data):
    regs = data.region.unique()
    qvals = pd.DataFrame({'region': regs, 'region_name': [r['name'] for r in
                                                          mcc.get_structure_tree().get_structures_by_acronym(
                                                              regs)]}).set_index('region').sort_index()

    for strain_name, strain in {"BL6": 'C57BL/6J', "CD1": 'FVB.CD1(ICR)', "ALL": None}.items():
        strain_data = data
        if strain is not None:
            strain_data = strain_data[strain_data.strain == strain]
        males = strain_data[strain_data.gender == 'M']
        females = strain_data[strain_data.gender == 'F']
        male_groups = males.groupby('region')
        female_groups = females.groupby('region')
        for param in ['count3d', 'density3d', 'coverage', "volume"]:
            male_groups_data = male_groups[to_col_name(strain_data, param)]
            female_groups_data = female_groups[to_col_name(strain_data, param)]
            male_data = male_groups_data.apply(np.array).sort_index()
            female_data = female_groups_data.apply(np.array).sort_index()
            qvals[f'{strain_name}_{param}_male_means'] = male_groups_data.median().sort_index()
            qvals[f'{strain_name}_{param}_female_means'] = female_groups_data.median().sort_index()
            for method_name, method in {'t-test': scipy.stats.ttest_ind, 'ranksum': scipy.stats.ranksums}.items():
                local_data = pd.DataFrame({'male': male_data, 'female': female_data}).sort_index()
                regs, pvals = zip(*[(t.Index, method(t.male, t.female).pvalue) for t in local_data.itertuples()])
                rejected, pval_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.1)
                pvals = pd.DataFrame({'region': regs, 'pval': pval_corrected, 'rejected': rejected}).set_index(
                    'region').sort_index()
                pvals['sign'] = np.sign(
                    qvals[f'{strain_name}_{param}_male_means'] - qvals[f'{strain_name}_{param}_female_means'])
                qvals[f"{strain_name}_male_vs_female_{param}_{method_name}_rejected"] = pvals.rejected
                qvals[f"{strain_name}_male_vs_female_{param}_{method_name}_qval"] = pvals.pval * pvals.sign
                qvals[f"{strain_name}_male_vs_female_{param}_{method_name}_log10(qval)"] = -np.log10(pvals.pval) * pvals.sign

        for gender, gender_groups in {'male': male_groups, 'female': female_groups}.items():
            for param in ['count', 'density', 'region_area']:
                left_data = gender_groups[to_col_name(strain_data, f'{param}_left')].apply(np.array).sort_index()
                right_data = gender_groups[to_col_name(strain_data, f'{param}_right')].apply(np.array).sort_index()
                qvals[f'{strain_name}_{param}_{gender}_left_means'] = gender_groups[to_col_name(strain_data,
                                                                                         f'{param}_left')].median().sort_index()
                qvals[f'{strain_name}_{param}_{gender}_right_means'] = gender_groups[to_col_name(strain_data,
                                                                                          f'{param}_right')].median().sort_index()
                for method_name, method in {'t-test': scipy.stats.ttest_ind, 'ranksum': scipy.stats.ranksums}.items():
                    local_data = pd.DataFrame({'left': left_data, 'right': right_data}).sort_index()
                    regs, pvals = zip(*[(t.Index, method(t.left, t.right).pvalue) for t in local_data.itertuples()])
                    rejected, pval_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.1)
                    pvals = pd.DataFrame({'region': regs, 'pval': pval_corrected, 'rejected': rejected}).set_index(
                        'region').sort_index()
                    pvals['sign'] = np.sign(
                        qvals[f'{strain_name}_{param}_{gender}_left_means']
                        - qvals[f'{strain_name}_{param}_{gender}_right_means'])
                    qvals[f"{strain_name}_left_{param}_{gender}_rejected"] = rejected
                    qvals[f"{strain_name}_left_vs_right_{param}_{gender}_{method_name}_rejected"] = pvals.rejected
                    qvals[f"{strain_name}_left_vs_right_{param}_{gender}_{method_name}_qval"] = pvals.pval * pvals.sign
                    qvals[f"{strain_name}_left_vs_right_{param}_{gender}_{method_name}_log10(qval)"] = -np.log10(
                        pvals.pval) * pvals.sign

    return qvals


def plot_qval_vs_qval(qvals, strain, prefix, highlight_col, x_col, y_col, filename):
    regions = qvals.index.tolist()
    fig, ax = get_subplots()
    highlight = qvals[strain + '_' + prefix + "_" + highlight_col + '_qval'].to_numpy().copy()
    highlight_rejected = qvals[strain + '_' + prefix + "_" + highlight_col + '_rejected'].to_numpy()
    highlight[~highlight_rejected] = 1
    x = qvals[strain + '_' + prefix + "_" + x_col + '_log10(qval)'].to_numpy()
    y = qvals[strain + '_' + prefix + "_" + y_col + '_log10(qval)'].to_numpy()
    highlight = (np.abs(highlight) <= 0.01)
    plot_scatter(ax, x[~highlight], y[~highlight], label=f'{highlight_col} not significant')
    plot_annotations(ax, np.array(regions)[highlight], x[highlight], y[highlight], dot_color='r')
    plot_grid(ax, [0], [0])
    produce_figure(ax, fig, strain + '_' + prefix + '_' + filename)


def produce_volcano_plot(qvals, strain, prefix, x_col1, x_col2, y_col, filename, hide_rejected=False):
    fig, ax = get_subplots()
    if hide_rejected:
        qvals = qvals[qvals[strain + '_' + prefix + '_' + y_col + '_rejected']]
    regions = qvals.index.tolist()
    x = np.array(qvals[strain + '_' + x_col1] - qvals[strain + '_' + x_col2]) / np.minimum(qvals[strain + '_' + x_col1].to_numpy(),
                                                                                           qvals[strain + '_' + x_col2].to_numpy())
    y = np.abs(np.array(qvals[strain + '_' + prefix + '_' + y_col + '_log10(qval)']))
    plot_scatter(ax, x, y)
    ann = np.where((np.abs(x) > 0.05) & (y > 2))
    plot_annotations(ax, np.array(regions)[ann], x[ann], y[ann], dot_color='r')
    plot_grid(ax, horiz_lines=[2], vert_lines=[0.05, -0.05])
    produce_figure(ax, fig, strain + '_' + prefix + '_' + filename)


def figure3(data):
    data = data.join(experiments[['strain', 'gender']], on='experiment_id')

    qvals = generate_qval_data(data)
    strains = ['BL6', 'CD1', 'ALL']
    genders = ['male', 'female']

    for strain in strains:
        produce_volcano_plot(qvals, strain, 'male_vs_female',
                             f'volume_male_means',
                             f'volume_female_means',
                             f'volume_ranksum',
                             f'volume_qval_vs_normalized_M-F')

        for param, highlight_param in [('count3d', 'density3d'), ('density3d', 'count3d')]:
            produce_volcano_plot(qvals, strain, 'male_vs_female',
                                 f'{param}_male_means',
                                 f'{param}_female_means',
                                 f'{param}_ranksum',
                                 f'{param}_qval_vs_normalized_M-F')
            plot_qval_vs_qval(qvals, strain, 'male_vs_female',
                              f'{highlight_param}_ranksum',
                              f'{param}_ranksum',
                              'volume_ranksum',
                              f'{param}_qval_vs_volume_qval')

        for (param, highlight_param), gender in itertools.product([('count', 'density'), ('density', 'count')],
                                                                  genders):
            produce_volcano_plot(qvals, strain, 'left_vs_right',
                                 f'{param}_{gender}_left_means',
                                 f'{param}_{gender}_right_means',
                                 f'{param}_{gender}_ranksum',
                                 f"{param}_{gender}_qval_vs_normalized_M-F")
            plot_qval_vs_qval(qvals, strain, 'left_vs_right',
                              f'{highlight_param}_{gender}_ranksum',
                              f'{param}_{gender}_ranksum',
                              f'region_area_{gender}_ranksum',
                              f"{param}_{gender}_qval_vs_volume_qval")

    for param in ['count3d', 'density3d', 'volume']:
        x = qvals[f'BL6_{param}_male_means'] - qvals[f'BL6_{param}_female_means']
        y = qvals[f'ALL_{param}_male_means'] - qvals[f'ALL_{param}_female_means']
        fig, ax = get_subplots()
        plot_scatter(ax, x, y)
        ax.set_xscale('log')
        ax.set_yscale('log')
        produce_figure(ax, fig, f'{param}_BL6_vs_ALL_M-F')



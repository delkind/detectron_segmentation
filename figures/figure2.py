import math

import numpy as np
import seaborn as sns

from figures.clean_data import plot_symmetry_score
from figures.util import main_regions, produce_figure, mcc, produce_plot_data, get_subplots, fig_config, plot_scatter, \
    plot_annotations, density_scatter, plot_grid


def plot_observed_vs_technical(data):
    res = produce_log_lr_and_count(data)
    regions = list(res['count'].keys())

    x = np.array([v for v in [res['count'][e] for e in regions]]).mean(axis=1)
    y = np.array([v for v in [res['l/r'][e] for e in regions]])
    y = np.sqrt(np.power(y, 2).sum(axis=1)) / y.shape[1]
    z = np.array([res['count'][e] for e in regions]).std(axis=1)
    fig, ax = get_subplots()

    for region, structs in main_regions.items():
        indices = np.where(np.isin(regions, structs))
        plot_scatter(ax, x[indices], y[indices], label=f'diff_{region}')
        plot_scatter(ax, x[indices], z[indices], label=f'std_{region}')
    produce_figure(ax, fig, fpath="observed_vs_technical", xlabel='mean', legend=True)

    fig, ax = get_subplots()
    for region, structs in main_regions.items():
        indices = np.where(np.isin(regions, structs))
        ax.scatter(x[indices], (z - y)[indices], s=fig_config['dot_size'], label=region)

    ann = np.where(y > 1)
    ann_x = x[ann]
    ann_y = (z-y)[ann]
    ann_text = np.array(regions)[ann]

    plot_annotations(ax, ann_text, ann_x, ann_y)
    produce_figure(ax, fig, fpath="observed_vs_technical_diff", xlabel='mean', ylabel='std-diff', legend=True)

    fig, ax = get_subplots()
    for region, structs in main_regions.items():
        indices = np.where(np.isin(regions, structs))
        ax.scatter(x[indices], (z/y)[indices], s=fig_config['dot_size'], label=region)

    # for i, key in enumerate(regions):
    #     y_coord = (y/z)[i]
    #     x_coord = x[i]
    #     if y_coord > 1.0:
    #         ax.annotate(regions[i], (x_coord, y_coord), fontsize='x-small')

    produce_figure(ax, fig, fpath="observed_vs_technical_ratio", xlabel='mean', ylabel='std/diff', legend=True)


def plot_logcount_vs_logdiff(data, remove_outliers):
    res = produce_plot_data(data,
                            {
                                'left': lambda s: s['count_left'],
                                'right': lambda s: s['count_right'],
                                'count': lambda s: s['count_left'] + s['count_right']
                            },
                            {
                                'left': np.array,
                                'right': np.array,
                                'count': np.array
                            })
    regions = list(res['count'].keys())
    x = np.array([np.log10(res['count'][r] + 1) for r in regions]).flatten()
    y = np.array([np.log10(res['left'][r] + 1) - np.log10(res['right'][r] + 1) for r in regions]).flatten()
    if remove_outliers:
        lower_threshold = np.percentile(y, 1)
        upper_threshold = np.percentile(y, 99)
        show = (y > lower_threshold) & (y < upper_threshold)
        y = y[show]
        x = x[show]

    fig, ax = get_subplots()
    density_scatter(ax, x, y)
    produce_figure(ax, fig, fpath="logcount_vs_logdiff", xlabel='logcount', ylabel='log(L-R)', logscale=False,
                   format_xticks=False, format_yticks=False)


def plot_count_vs_diff(data):
    res = produce_plot_data(data,
                            {
                                'l-r': lambda s: abs(s['count_left'] - s['count_right']),
                                'count': lambda s: s['count']
                            },
                            {
                                'l-r': np.array,
                                'count': np.array
                            })
    regions = list(res['count'].keys())
    x = np.array([np.mean([e for e in res['count'][r]]) for r in regions])
    y = np.array([np.mean([e for e in res['l-r'][r]]) for r in regions])
    # z = np.array([np.std([e for e in res['count'][r]]) for r in regions])

    fig, ax = get_subplots()
    for region, structs in main_regions.items():
        indices = np.where(np.isin(regions, structs))
        plot_scatter(ax, x[indices], y[indices], label=region)
    produce_figure(ax, fig, fpath="count_vs_diff", xlabel='mean', ylabel='|l-r|', logscale=True, legend=True)


def plot_count_vs_brightness(valid_data):
    regions = list(list(valid_data.values())[0].keys())
    count = np.log10(np.array([[valid_data[e][r]['count'] for e in valid_data.keys()] for r in regions]).flatten())
    brightness = np.array([[valid_data[e][r]['brightness']['mean'] for e in valid_data.keys()] for r in regions]).flatten()
    fig, ax = get_subplots()
    density_scatter(ax, count, brightness)
    produce_figure(ax, fig, "count_vs_brightness", 'logcount', 'brightness')


def produce_log_lr_and_count(data):
    res = produce_plot_data(data,
                            {
                                'l/r': lambda s: (s['count_left'] + 1)/(s['count_right'] + 1),
                                'count': lambda s: math.log10(s['count'] + 1),
                            },
                            {
                                'l/r': lambda l: np.abs(np.log10(l)),
                                'count': np.array,
                            })
    return res


def figure2(data, remove_outliers=False):
    plot_logcount_vs_logdiff(data, remove_outliers)
    plot_count_vs_diff(data)
    plot_observed_vs_technical(data)
    plot_symmetry_score(data)
    plot_count_vs_brightness(data)

    # plot_correlation_histogram(valid_data, 'brightness', 'count')
    # plot_correlation_histogram(valid_data, 'brightness', 'density')
    # plot_correlation_histogram(valid_data, "volume", "count")
    # plot_correlation_histogram(valid_data, "density", "count")
    # plot_correlation_histogram(valid_data, "coverage", "count")


# import os
# from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
# analyzed = set(os.listdir("output/full_brain/analyzed"))
# mcc = MouseConnectivityCache(manifest_file='mouse_connectivity/mouse_connectivity_manifest.json', resolution=25)
# cd1_exps = mcc.get_experiments(dataframe=True)
# cd1_exps = cd1_exps[cd1_exps.strain == 'FVB.CD1(ICR)']
# males = cd1_exps[cd1_exps.gender == 'M']
# females = cd1_exps[cd1_exps.gender == 'F']
# male_ids = set([str(i) for i in males.id.tolist()])
# female_ids = set([str(i) for i in females.id.tolist()])
#
# os.makedirs("output/cd1_males/input")
# for i in male_ids - analyzed:
#     os.rename(f"output/full_brain/input/{i}", f"output/cd1_males/input/{i}")
#
# os.makedirs("output/cd1_females/input")
# for i in female_ids - analyzed:
#     os.rename(f"output/full_brain/input/{i}", f"output/cd1_females/input/{i}")
import numpy as np
import seaborn as sns
import pandas as pd

from figures.util import get_subplots, plot_grid, produce_figure, extract_data, plot_scatter, plot_annotations, \
    fig_config, mcc


def brain_count_cdf(data):
    counts = [e['grey']['count3d'] for e in data.values()]
    fig, ax = get_subplots()
    sns.histplot(x=counts, ax=ax, cumulative=True, stat='density', bins=50)
    plot_grid(ax, vert_lines=[], horiz_lines=[0.5])
    produce_figure(ax, fig, "brain_count_cdf")


def param_ranked_list(data, param):
    exps = list(data.keys())
    regions = np.array(list(data[exps[0]].keys()))
    levels = np.array([len(s['structure_id_path'])
                       for s in mcc.get_structure_tree().get_structures_by_acronym(regions)])
    regions = regions[levels == 8]
    param_data = np.array([np.mean([extract_data(data[e][r], param) for e in exps]) for r in regions])
    idx = np.argsort(param_data)[::-1]

    df = pd.DataFrame({
        'region': regions[idx],
        param: param_data[idx],
    })

    fig, ax = get_subplots()
    sns.set_color_codes("muted")
    sns.scatterplot(data=df, y=param, x='region', color='b')
    plot_annotations(ax,
                     np.concatenate([regions[idx][:10], regions[idx][-10:]]),
                     np.concatenate([np.arange(len(param_data))[:10], np.arange(len(param_data))[-10:]]),
                     np.concatenate([param_data[idx][:10], param_data[idx][-10:]]))
    ax.tick_params(axis='y', length=fig_config['tick_length'], labelsize=fig_config['tick_labelsize'])
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    produce_figure(ax, fig, f"{param}_ranked_list", format_yticks=False)


def count_pie_chart(data, level, descend_from='grey'):
    exps = list(data.keys())
    regions = np.array(list(data[exps[0]].keys()))
    regions, colors = regions_by_level(level, regions, descend_from)
    counts = np.array([np.median([extract_data(data[e][r], 'count3d') for e in exps]) for r in regions])

    df = pd.DataFrame({
        'region': regions,
        'count3d': counts,
    })

    fig, ax = get_subplots()
    ax.pie(x=counts, autopct="%.1f%%", explode=[0.05] * len(counts), labels=regions, pctdistance=0.5, colors=colors)
    produce_figure(ax, fig, f"{descend_from}_level_{level}_counts_chart", format_yticks=False)


def regions_by_level(level, regions, descend_from):
    descend_from = mcc.get_structure_tree().get_structures_by_acronym([descend_from])[0]['id']
    paths = {r: s['structure_id_path'] for r, s in
             zip(regions, mcc.get_structure_tree().get_structures_by_acronym(regions))}
    colors = {r: np.array(s['rgb_triplet']) / 255 for r, s in
              zip(regions, mcc.get_structure_tree().get_structures_by_acronym(regions))}
    regions = np.array(list(filter(lambda r: descend_from in paths[r], regions)))
    levels = np.array([len(paths[r]) for r in regions])
    regions = regions[levels == level]

    return regions, [colors[r] for r in regions]


def figure1(data):
    for param in ['area', 'density', 'coverage']:
        param_ranked_list(data, param)
    for level in [3, 4, 5, 6, 7, 8, 9]:
        count_pie_chart(data, level)
    count_pie_chart(data, 9, "Isocortex")

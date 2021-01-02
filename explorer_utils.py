from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def reject_outliers(data):
    if len(data) > 2:
        clean_data = data[data < np.percentile(data, 99)]
        if len(clean_data > 0):
            return clean_data
        else:
            return data
    else:
        return data


def hist(ax, d, bins, **kwargs):
    d = reject_outliers(d)
    if len(d) > 2:
        _, hst = np.histogram(d, bins=bins)
        ax.plot(hst, stats.gaussian_kde(d)(hst), **kwargs)
    else:
        ax.hist([], bins=bins, **kwargs)


def violinplot(ax, structure_data, col):
    def set_axis_style(ax, labels, xlabel):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation='vertical')
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel(xlabel)

    labels = [f'{struct} {attr}' for struct, attrs in structure_data.items() for attr in attrs.keys()]
    ax.violinplot([reject_outliers(structure_data[struct][attr][col]) for struct, attrs in
                   structure_data.items() for attr in attrs.keys()], showmedians=True)
    set_axis_style(ax, labels, col.capitalize())


def prepare_structure_data(data, structure_tree):
    def fill_data(attr, dt, struct, structure_data):
        if len(dt) > 0:
            for col in ['density', 'area']:
                structure_data[struct][attr][col] = dt[col].to_numpy()

    def nested_defaultdict():
        return defaultdict(nested_defaultdict)

    structure_data = nested_defaultdict()

    for s in ['Field CA1', 'Field CA2', 'Field CA3']:
        data_struct = data[data.structure_id == structure_tree.get_structures_by_name([s])[0]['id']]
        attr_data = dict()
        attr_data['Dense'] = data_struct[data_struct.dense == True]
        attr_data['Sparse'] = data_struct[data_struct.dense == False]

        for a, d in attr_data.items():
            fill_data(a, d, s, structure_data)

    for a in [632, 10703, 10704]:
        data_struct = data[data.structure_id == a]
        fill_data(str(a), data_struct, 'DG', structure_data)

    return structure_data


def display_totals(ax_density, ax_area, bins, data):
    hist(ax_density, reject_outliers(data.density.to_numpy()), bins=bins)
    ax_density.set_title("Density")
    hist(ax_area, reject_outliers(data.area.to_numpy()), bins=bins)
    ax_area.set_title("Area")


def plot_section_histograms(data, structure_tree, bins=50):
    fig, axs = plt.subplots(2, 5, constrained_layout=True, figsize=(25, 10))
    axs = axs.flatten()

    display_totals(axs[0], axs[5], bins, data)
    structure_data = prepare_structure_data(data, structure_tree)

    for i, struct in enumerate(structure_data.keys()):
        total_density = np.array([], dtype=float)
        total_area = np.array([], dtype=float)
        for attr in structure_data[struct].keys():
            data_slice = structure_data[struct][attr]
            total_density = np.concatenate((total_density, data_slice['density']))
            total_area = np.concatenate((total_area, data_slice['area']))
            hist(axs[i + 1], data_slice['density'], bins=bins, alpha=0.5, label=attr)
            hist(axs[i + 6], data_slice['area'], bins=bins, alpha=0.5, label=attr)

        hist(axs[i + 1], total_density, bins=bins, alpha=0.5, label="Total")
        hist(axs[i + 6], total_area, bins=bins, alpha=0.5, label="Total")

        axs[i + 1].legend()
        axs[i + 1].set_title(struct + " density")
        axs[i + 6].legend()
        axs[i + 6].set_title(struct + " area")

    plt.show()


def plot_section_violin_diagram(data, structure_tree, thumb=None, bins=50):
    fig, axs = plt.subplots(1, 4 + int(thumb is not None), constrained_layout=True,
                            figsize=(18 + 4 * int(thumb is not None), 5))

    display_totals(axs[0], axs[1], bins, data)

    structure_data = prepare_structure_data(data, structure_tree)

    violinplot(axs[2], structure_data, 'area')
    violinplot(axs[3], structure_data, 'density')

    if thumb is not None:
        axs[4].imshow(thumb)
        axs[4].axis('off')
    plt.show()


def test():
    import os
    import random
    import pandas as pd

    input_dir = 'output/hippo_exp/analyzed'
    experiment_ids = os.listdir(input_dir)
    experiment_id = random.choice(experiment_ids)
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    mcc = MouseConnectivityCache(manifest_file=f'mouse_connectivity/mouse_connectivity_manifest.json',
                                 resolution=25)
    directory = f"{input_dir}/{experiment_id}"
    full_data = pd.read_csv(f"{directory}/celldata-{experiment_id}.csv")
    full_data_mtime = os.path.getmtime(f"{directory}/celldata-{experiment_id}.csv")
    plot_section_histograms(full_data, mcc.get_structure_tree())
    plot_section_violin_diagram(full_data, mcc.get_structure_tree())


if __name__ == '__main__':
    test()

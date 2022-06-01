import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
import seaborn as sns

from experiment_images_predictor import extract_predictions


def reject_outliers(data):
    if len(data) > 2:
        clean_data = data[data < np.percentile(data, 99)]
        if len(clean_data > 2):
            return clean_data
        else:
            return data
    else:
        return data


def hist(ax, d, bins, raw_hist=False, median=False, **kwargs):
    m = np.median(d)
    # d = reject_outliers(d)
    if len(d) > 2:
        _, hst = np.histogram(d, bins=bins)
        line = ax.plot(hst, stats.gaussian_kde(d)(hst), **kwargs)
        if raw_hist:
            ax.hist(d, bins=bins, histtype=u'step', density=True, color=line[0].get_color(), linestyle=':')
        if median:
            ax.axvline(x=m, color=line[0].get_color(), linestyle='--')
        # sns.histplot(d, bins=bins, stat='density', element='step', kde=True, fill=False,
        #              line_kws=dict(linestyle='--', alpha=0.5),
        #              **kwargs)
    else:
        ax.hist([], bins=bins, **kwargs)


def scatter(ax, x, y, xlabel, ylabel, show_identity=False, margins=None, **kwargs):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x, y, **kwargs)
    if show_identity or margins is not None:
        lims = (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        xpoints = ypoints = ax.get_xlim()
        if show_identity:
            ax.plot(xpoints, ypoints, linestyle='-.', color='r', lw=1, scalex=False, scaley=False)

        if margins is not None:
            ax.plot(xpoints, np.array(ypoints) - abs(ypoints[0] - ypoints[1]) * margins, linestyle='-.', color='r', lw=1,
                    scalex=False, scaley=False)
            ax.plot(xpoints, np.array(ypoints) + abs(ypoints[0] - ypoints[1]) * margins, linestyle='-.', color='r', lw=1,
                    scalex=False, scaley=False)



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
            for col in ['coverage', 'area']:
                structure_data[struct][attr][col] = dt[col].to_numpy()

    def nested_defaultdict():
        return defaultdict(nested_defaultdict)

    structure_data = nested_defaultdict()

    for s in ['Field CA1', 'Field CA2', 'Field CA3']:
        relevant_structs = structure_tree.descendant_ids([structure_tree.get_structures_by_name([s])[0]['id']])[0]
        data_struct = data[data.structure_id.isin(relevant_structs)]
        attr_data = dict()
        attr_data['Dense'] = data_struct[data_struct.dense == True]
        attr_data['Sparse'] = data_struct[data_struct.dense == False]

        for a, d in attr_data.items():
            fill_data(a, d, s, structure_data)

    for a in [632, 10703, 10704]:
        data_struct = data[data.structure_id == a]
        fill_data(str(a), data_struct, 'DG', structure_data)

    return structure_data


def display_totals(ax_coverage, ax_area, bins, data):
    hist(ax_coverage, reject_outliers(data.coverage.to_numpy()), bins=bins)
    ax_coverage.set_title("coverage")
    hist(ax_area, reject_outliers(data.area.to_numpy()), bins=bins)
    ax_area.set_title("Area")


def plot_section_histograms(data, structure_tree, bins=50):
    fig, axs = plt.subplots(2, 5, constrained_layout=True, figsize=(25, 10))
    axs = axs.flatten()

    display_totals(axs[0], axs[5], bins, data)
    structure_data = prepare_structure_data(data, structure_tree)

    for i, struct in enumerate(structure_data.keys()):
        total_coverage = np.array([], dtype=float)
        total_area = np.array([], dtype=float)
        for attr in structure_data[struct].keys():
            data_slice = structure_data[struct][attr]
            total_coverage = np.concatenate((total_coverage, data_slice['coverage']))
            total_area = np.concatenate((total_area, data_slice['area']))
            hist(axs[i + 1], data_slice['coverage'], bins=bins, alpha=0.5, label=attr)
            hist(axs[i + 6], data_slice['area'], bins=bins, alpha=0.5, label=attr)

        hist(axs[i + 1], total_coverage, bins=bins, alpha=0.5, label="Total")
        hist(axs[i + 6], total_area, bins=bins, alpha=0.5, label="Total")

        axs[i + 1].legend()
        axs[i + 1].set_title(struct + " coverage")
        axs[i + 6].legend()
        axs[i + 6].set_title(struct + " area")

    plt.show()


def plot_section_violin_diagram(data, structure_tree, thumb=None, bins=50):
    fig, axs = plt.subplots(1, 4 + int(thumb is not None), constrained_layout=True,
                            figsize=(18 + 4 * int(thumb is not None), 5))

    display_totals(axs[0], axs[1], bins, data)

    structure_data = prepare_structure_data(data, structure_tree)

    violinplot(axs[2], structure_data, 'area')
    violinplot(axs[3], structure_data, 'coverage')

    if thumb is not None:
        axs[4].imshow(thumb)
        axs[4].axis('off')
    plt.show()


def retrieve_nested_path(data, path):
    for p in path:
        data = data[p]

    return data


class DataFramesHolder(object):
    def __init__(self, data_dir, lazy=True, cache=True):
        self.cache = cache
        self.lazy = lazy
        self.data_dir = data_dir
        self.data = dict()

    def load_data(self, e):
        e = int(e)
        df = self.data.get(e, pd.read_parquet(f'{self.data_dir}/{e}/celldata-{e}.parquet'))
        if self.cache:
            self.data[e] = df
        return df

    def __getitem__(self, item):
        if isinstance(item, list):
            return pd.concat([self.load_data(e) for e in item])
        else:
            return self.load_data(item)


def is_file_up_to_date(path, base_time):
    return os.path.isfile(path) and os.path.getmtime(path) > base_time


def init_model(model_path, device, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return DefaultPredictor(cfg)


def predict_crop(crop, model):
    import cv2
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    outputs = model(crop)
    polygons, mask = extract_predictions(outputs["instances"].to("cpu"))
    cv2.polylines(crop, polygons, isClosed=True, color=(0, 255, 0))
    return crop


def test():
    import os
    import random

    input_dir = 'output/hippo_exp/analyzed'
    experiment_ids = os.listdir(input_dir)
    experiment_id = random.choice(experiment_ids)
    holder = DataFramesHolder(input_dir)
    pass


if __name__ == '__main__':
    test()

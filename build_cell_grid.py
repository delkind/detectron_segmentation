import pickle
import sys
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from annotate_cell_data import produce_patch_collection, plot_patch_collection
from rect import Rect

NROWS = 10
NCOLS = 7
FIG_SCALE = 20


def build_section(params):
    section, experiment, experiment_dir, bboxes, seg_data = params
    return produce_patch_collection(bboxes, seg_data, experiment_dir, experiment, section)


def build_cell_grid(experiment, data_dir, seg_data_dir, output_path):
    experiment_dir = f"{data_dir}/{experiment}"
    bboxes = pickle.load(open(f"{experiment_dir}/bboxes.pickle", "rb"))
    seg_data = np.load(f'{seg_data_dir}/{experiment}/{experiment}-sections.npz')['arr_0']

    params = []
    for section in list(bboxes.keys())[::-2]:
        params.append((section,
                       experiment,
                       experiment_dir,
                       bboxes,
                       seg_data))

    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(NCOLS * 3 * FIG_SCALE, NROWS * 2 * FIG_SCALE), dpi=125)
    for a in ax.flatten():
        a.axis('off')

    pool = Pool(cpu_count() // 2)
    patches = list(tqdm(pool.imap(build_section, params), "Building sections", total=len(params)))

    max_w = max(map(lambda t: t[0].w, patches))
    max_h = max(map(lambda t: t[0].h, patches))

    brain_bbox = Rect(0, 0, max_w, max_h)
    for i, result in enumerate(tqdm(patches, "Plotting sections", total=len(patches))):
        _, patches = result
        plot_patch_collection(ax.flatten()[i], brain_bbox, patches)

    print("Saving figure...")

    fig.savefig(output_path, dpi=125)
    plt.close()


if __name__ == '__main__':
    build_cell_grid(*(sys.argv[1:]))

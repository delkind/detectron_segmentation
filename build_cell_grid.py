import pickle
import sys
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from annotate_cell_data import produce_patch_collection, plot_patch_collection


def build_section(params):
    section, experiment, experiment_dir, bboxes, seg_data = params
    return produce_patch_collection(bboxes, seg_data, experiment_dir, experiment, section)


def build_cell_grid(experiment, data_dir, seg_data_dir, output_path, use_tqdm=False):
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

    if not use_tqdm:
        tqd = lambda p1, p2, total: p1
    else:
        tqd = tqdm

    pool = Pool(cpu_count() // 2)
    patches = list(tqd(pool.imap(build_section, params), "Building sections", total=len(params)))

    fig, ax = plt.subplots(nrows=10, ncols=7, figsize=(210, 200), dpi=25)
    for a in ax.flatten():
        a.axis('off')

    print(5)
    for i, result in enumerate(tqd(patches, "Plotting sections", total=len(patches))):
        brain_bbox, patches = result
        plot_patch_collection(ax.flatten()[i], brain_bbox, patches)

    print("Saving figure...")

    fig.savefig(output_path, dpi=25)
    plt.close()


# if __name__ == '__main__':
#     print(sys.argv)
#     build_cell_grid(*(sys.argv[1:]))

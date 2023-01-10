import pickle
import sys
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from annotate_cell_data import plot_patch_collection
from build_cell_grid import build_section
from rect import Rect

FIG_SCALE = 20


def build_cell_grid(experiment, data_dir, seg_data_dir, output_path):
    experiment_dir = f"{data_dir}/{experiment}"
    bboxes = pickle.load(open(f"{experiment_dir}/bboxes.pickle", "rb"))
    seg_data = np.load(f'{seg_data_dir}/{experiment}/{experiment}-sections.npz')['arr_0']

    params = []
    for section in list(bboxes.keys())[::-2][9:][::7]:
        params.append((section,
                       experiment,
                       experiment_dir,
                       bboxes,
                       seg_data))

    pool = Pool(cpu_count() // 2)
    patches = list(tqdm(pool.imap(build_section, params), "Building sections", total=len(params)))
    # patches = list(tqdm(map(build_section, params), "Building sections", total=len(params)))

    max_w = max(map(lambda t: t[0].w, patches))
    max_h = max(map(lambda t: t[0].h, patches))

    brain_bbox = Rect(0, 0, max_w, max_h)

    for i, result in enumerate(tqdm(patches, "Plotting sections", total=len(patches))):
        fig, ax = plt.subplots(figsize=(3 * FIG_SCALE, 2 * FIG_SCALE), dpi=125)
        _, patches, colors = result
        plot_patch_collection(ax, brain_bbox, patches)
        fig.savefig(output_path[:output_path.rfind('.')] + f'-{i}' + output_path[output_path.rfind('.'):], dpi=125)
        plt.close()


if __name__ == '__main__':
    build_cell_grid(*(sys.argv[1:]))

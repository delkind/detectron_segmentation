import ast
import math
import os
import pickle
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager

unique_colors = [
    (255, 0, 0)[::-1],
    (255, 255, 0)[::-1],
    (0, 234, 255)[::-1],
    (170, 0, 255)[::-1],
    (255, 127, 0)[::-1],
    (191, 255, 0)[::-1],
    (0, 149, 255)[::-1],
    (255, 0, 170)[::-1],
    (255, 212, 0)[::-1],
    (106, 255, 0)[::-1],
    (0, 64, 255)[::-1],
    (237, 185, 185)[::-1],
    (185, 215, 237)[::-1],
    (231, 233, 185)[::-1],
    (220, 185, 237)[::-1],
    (185, 237, 224)[::-1],
    (143, 35, 35)[::-1],
    (35, 98, 143)[::-1],
    (143, 106, 35)[::-1],
    (107, 35, 143)[::-1],
    (79, 143, 35)[::-1],
    (0, 0, 0)[::-1],
    (115, 115, 115)[::-1],
    (204, 204, 204)[::-1],
]


class ExperimentDataAnnotator(object):
    def __init__(self, experiment_id, directory, logger):
        self.logger = logger
        self.directory = directory
        self.experiment_id = experiment_id
        with open(f'{self.directory}/bboxes.pickle', "rb") as f:
            bboxes = pickle.load(f)
        self.bboxes = {k: v for k, v in bboxes.items() if v}
        self.celldata = pd.read_csv(f'{self.directory}/celldata-{self.experiment_id}.csv')
        self.tile_dim = int(math.ceil(math.sqrt(len(self.bboxes))))

    @staticmethod
    def generate_colormap(N):
        if N < 2:
            return np.array([0.9, 0, 0, 1])
        arr = np.arange(N) / N
        arr = arr.reshape(N, 1).T.reshape(-1)
        ret = matplotlib.cm.hsv(arr)
        n = ret[:, 3].size
        a = n // 2
        b = n - a
        for i in range(3):
            ret[0:n // 2, i] *= np.arange(0.2, 1, 0.8 / a)
        ret[n // 2:, 3] *= np.arange(1, 0.1, -0.9 / b)
        #     print(ret)
        return ret

    def transparent_cmap(cmap, N=255):
        mycmap = cmap
        mycmap._init()
        mycmap._lut[0, -1] = 0
        return mycmap

    def process(self):
        self.create_images()
        self.create_tiles(placer=self.place_heatmap, name='heatmaps', zoom=1, binsize=5)
        self.create_tiles(placer=self.place_patches, name='patches', zoom=4, gridsize=5)

    def create_images(self):
        self.logger.info(f"Experiment {self.experiment_id}: Creating annotated images...")
        for section in self.bboxes.keys():
            self.create_section_image(section)

    def create_section_image(self, section):
        section_celldata = self.celldata[self.celldata.section == section]
        section_numbers = section_celldata.structure_id.to_numpy()
        section_modifiers = -(section_celldata.pyramidal.to_numpy().astype(int))
        section_modifiers[section_modifiers == 0] = 1
        section_numbers *= section_modifiers
        coords = (np.stack((section_celldata.centroid_x.to_numpy() // 64,
                            section_celldata.centroid_y.to_numpy() // 64,
                            section_numbers)).swapaxes(0, 1)).astype(int)
        coords = list(zip(*list(zip(*coords.tolist()))))
        coords = {(x, y): v for x, y, v in coords}
        unique_numbers = np.unique(section_numbers).tolist()
        colors = {v: unique_colors[i % len(unique_colors)] for i, v in enumerate(unique_numbers)}

        thumb = cv2.imread(f"{self.directory}/thumbnail-{self.experiment_id}-{section}.jpg", cv2.IMREAD_GRAYSCALE)
        thumb = cv2.resize(thumb, (0, 0), fx=16, fy=16)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        for bbox in self.bboxes[section]:
            x, y, w, h = bbox.scale(64)
            image = cv2.imread(f'{self.directory}/full-{self.experiment_id}-{section}-{x}_{y}_{w}_{h}.jpg',
                               cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(f'{self.directory}/cellmask-{self.experiment_id}-{section}-{x}_{y}_{w}_{h}.png',
                              cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_img = np.zeros_like(mask)
            new_img = cv2.fillPoly(new_img, cnts, color=255)
            cnts, _ = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [cnt for cnt in cnts if cnt.shape[0] > 2]
            polygons = [Polygon((cnt.squeeze() + np.array([x, y])) // 64).centroid for cnt in cnts]
            polygons = [(int(p.x), int(p.y)) for p in polygons]
            cnts = {v: [cnt.squeeze() // 4 for i, cnt in enumerate(cnts) if coords.get(polygons[i], 0) == v]
                    for v in unique_numbers}
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for p in unique_numbers:
                cv2.polylines(image, cnts[p], color=colors[p], thickness=1, isClosed=True)
            thumb[y // 4: y // 4 + image.shape[0], x // 4: x // 4 + image.shape[1], :] = image
        cv2.imwrite(f"{self.directory}/annotated-{self.experiment_id}-{section}.jpg", thumb)

    def create_tiles(self, placer, name, zoom, **kwargs):
        fig, axs = plt.subplots(self.tile_dim, self.tile_dim, constrained_layout=True)
        fig.suptitle(self.experiment_id, fontsize=8)
        for ax in axs.flatten().tolist():
            ax.set_axis_off()

        for num, section in enumerate(sorted(list(self.bboxes.keys()))):
            self.logger.debug(f"Experiment {self.experiment_id}: creating {name} for section {section} "
                              f"({num + 1}/{len(self.bboxes.keys())})")
            ax = axs[num // self.tile_dim, num % self.tile_dim]
            ax.set_title(section, fontsize=6)
            labels, p = self.process_section(ax, kwargs, placer, section, zoom)
            self.decorate_section(ax, fig, labels, p)

        self.logger.info(f"Experiment {self.experiment_id}: Saving {name}...")
        plt.savefig(f"{self.directory}/{name}-{self.experiment_id}.pdf", dpi=2400)
        plt.close()

    def process_section(self, ax, kwargs, placer, section, zoom):
        section_celldata = self.celldata[self.celldata.section == section]
        thumb = Image.open(f'{self.directory}/thumbnail-{self.experiment_id}-{section}.jpg').convert('LA')
        thumb = thumb.resize((thumb.size[0] * zoom, thumb.size[1] * zoom))
        labels, p = placer(ax=ax, thumb=thumb, section_celldata=section_celldata, zoom=zoom, **kwargs)
        return labels, p

    @staticmethod
    def decorate_section(ax, fig, labels, p):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        bar = fig.colorbar(p, cax=cax, ax=ax)
        bar.ax.tick_params(length=1, pad=0, labelsize=2)
        if labels:
            bar.set_ticks(sorted(list(labels.values())))
            bar.ax.set_yticklabels(labels)

    @staticmethod
    def place_patches(ax, thumb, gridsize, section_celldata, zoom, radius=3, colname='pyramidal'):
        ax.imshow(thumb)
        structs = np.unique(section_celldata.structure_id.to_numpy() + section_celldata[colname].to_numpy()).tolist()
        struct_counts = {s: len(section_celldata[(section_celldata.structure_id + section_celldata[colname]) == s]) for
                         s in structs}
        cmap = ListedColormap(ExperimentDataAnnotator.generate_colormap(len(structs)))
        coords = np.stack((section_celldata.centroid_x.to_numpy() / (64 / zoom * gridsize),
                           section_celldata.centroid_y.to_numpy() / (64 / zoom * gridsize),
                           section_celldata.structure_id.to_numpy() + section_celldata[colname].to_numpy())).astype(
            int).tolist()
        coords = sorted(list({(x, y, s) for x, y, s in zip(*coords)}), key=lambda t: struct_counts[t[2]], reverse=True)
        patches = [Circle((x * gridsize, y * gridsize), radius) for x, y, _ in coords]
        colors = [c for _, _, c in coords]
        p = PatchCollection(patches, cmap=cmap, alpha=1.0)
        p.set_array((np.array([structs.index(s) for s in colors]) + 1))
        ax.add_collection(p)
        labels = {str(s): (i + 1) for i, s in enumerate(structs)}
        return labels, p

    @staticmethod
    def place_heatmap(ax, thumb, section_celldata, zoom, binsize):
        heatmap = np.zeros((thumb.size[1], thumb.size[0]), dtype=float)
        x = (section_celldata.centroid_x.to_numpy() // 64).astype(int)
        y = (section_celldata.centroid_y.to_numpy() // 64).astype(int)
        heatmap[y, x] = section_celldata.density.to_numpy()
        ax.imshow(thumb)
        ax.imshow(heatmap, cmap=ExperimentDataAnnotator.transparent_cmap(plt.get_cmap('hot')))
        return {}, matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(heatmap.min(), heatmap.max()),
            cmap=plt.get_cmap('hot'))


class CellProcessor(DirWatcher):
    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, structs, connectivity_dir,
                 _processor_number):
        super().__init__(input_dir, process_dir, output_dir, f'cell-processor-{_processor_number}')
        self.structure_ids = structs
        self.brain_seg_data_dir = structure_map_dir
        self.source_dir = input_dir
        self.output_dir = output_dir

    def process_item(self, item, directory):
        experiment = ExperimentDataAnnotator(item, directory, self.logger)
        experiment.process()


class ExperimentCellAnalyzerTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment cell data analyzer")

    def prepare_input(self, connectivity_dir, **kwargs):
        pass

    def execute_task(self, structs, structure_map_dir, **kwargs):
        analyzer = CellProcessor(structs=ast.literal_eval(structs), structure_map_dir=structure_map_dir, **kwargs)
        experiments = os.listdir(structure_map_dir)
        # analyzer.run_until_count(len(experiments))
        analyzer.run_until_count(len(experiments))


if __name__ == '__main__':
    task_mgr = ExperimentCellAnalyzerTaskManager()
    task_mgr.run()

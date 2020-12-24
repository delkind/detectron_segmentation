import ast
import functools
import operator as op
import os
import pickle
from collections import defaultdict
from functools import reduce

import cv2
import numpy as np
import pandas
import scipy.ndimage as ndi
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from shapely.geometry import Polygon
from sklearn.cluster import KMeans

from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager
from localize_brain import detect_brain


class ExperimentCellsProcessor(object):
    def __init__(self, mcc, experiment_id, directory, brain_seg_data_dir, parent_struct_id,
                 experiment_fields_to_save, details, logger, default_struct_id=997):
        self.experiment_fields_to_save = experiment_fields_to_save
        self.default_struct_id = default_struct_id
        self.parent_struct_id = parent_struct_id
        self.brain_seg_data_dir = brain_seg_data_dir
        self.directory = directory
        self.mcc = mcc
        self.id = experiment_id
        mapi = MouseConnectivityApi()
        self.details = {**details, **(mapi.get_experiment_detail(self.id)[0])}
        self.logger = logger
        self.subimages = {i['section_number']: i for i in self.details['sub_images']}
        self.seg_data = np.load(f'{self.brain_seg_data_dir}/{self.id}/{self.id}-sections.npz')['arr_0']
        self.structure_tree = self.mcc.get_structure_tree()
        self.structure_ids = self.get_structure_children()
        with open(f'{self.directory}/bboxes.pickle', "rb") as f:
            bboxes = pickle.load(f)
        self.bboxes = {k: v for k, v in bboxes.items() if v}

    def get_structure_children(self):
        structure_ids = self.structure_tree.descendant_ids(self.parent_struct_id)
        structure_ids = set(functools.reduce(op.add, structure_ids))
        return structure_ids

    def get_structure_mask(self, section):
        section_seg_data = self.seg_data[:, :, section]
        mask = np.isin(section_seg_data, self.structure_ids)
        mask = ndi.binary_closing(ndi.binary_fill_holes(mask).astype(np.int8)).astype(np.int8)
        return mask

    def calculate_densities(self, csv):
        csv['density'] = 0
        for section in sorted(np.unique(csv.section)):
            csv_section = csv[csv.section == section]
            self.logger.debug(f"Creating heatmap experiment {self.id} section {section}")
            image = np.zeros((self.seg_data.shape[0] * 64, self.seg_data.shape[1] * 64), dtype=np.int16)
            for bbox in self.bboxes.get(section, []):
                x, y, w, h = bbox.scale(64)
                cellmask = cv2.imread(f'{self.directory}/cellmask-{self.id}-{section}-{x}_{y}_{w}_{h}.png',
                                      cv2.IMREAD_GRAYSCALE)
                cnts, _ = cv2.findContours(cellmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                new_img = np.zeros_like(cellmask)
                new_img = cv2.fillPoly(new_img, cnts, color=1)
                image[y: y + cellmask.shape[0], x: x + cellmask.shape[1]] = new_img

            centroids_y = csv_section.centroid_y.to_numpy().astype(int)
            centroids_x = csv_section.centroid_x.to_numpy().astype(int)
            densities = np.array([image[centroids_y[i] - 32: centroids_y[i] + 32,
                                  centroids_x[i] - 32: centroids_x[i] + 32].sum() for i in range(len(centroids_y))])
            csv.loc[csv.section == section, 'density'] = densities / 4096

    def detect_pyramidal(self, csv):
        structures_including_pyramidal = [f'Field CA{i}' for i in range(1, 4)]
        dense_masks = defaultdict(dict)
        celldata_structs = csv[csv.structure.isin(structures_including_pyramidal)]
        relevant_sections = sorted(np.unique(celldata_structs.section.to_numpy()).tolist())

        for structure in structures_including_pyramidal:
            celldata_struct = celldata_structs.loc[csv.structure == structure]

            for section in relevant_sections:
                celldata_section = celldata_struct[celldata_struct.section == section]
                centroids_x = (celldata_section.centroid_x.to_numpy() // 64).astype(int)
                centroids_y = (celldata_section.centroid_y.to_numpy() // 64).astype(int)
                densities = celldata_section.density.to_numpy()
                dense_mask = np.zeros_like(self.seg_data[:, :, section])

                if densities.shape[0] > 2:
                    model = KMeans(n_clusters=2)
                    yhat = model.fit_predict(densities.reshape(-1, 1))
                    clusters = np.unique(yhat).tolist()
                    if len(clusters) == 1:
                        yhat = np.zeros_like(densities)
                        dense = 1
                    else:
                        dense = np.argmax([densities[yhat == 0].mean(), densities[yhat == 1].mean()])

                    # border = np.percentile(densities, 50)
                    dense_mask[centroids_y[yhat == dense], centroids_x[yhat == dense]] = 1
                    dense_mask = ndi.binary_closing(dense_mask, ndi.generate_binary_structure(2, 1), iterations=4)
                    dense_mask, comps = ndi.measurements.label(dense_mask)
                    if comps > 0:
                        sums = np.array([(dense_mask == i + 1).sum() for i in range(comps)])
                        comps = np.argwhere((sums > 10) & ((sums.max() / sums) < 5)).flatten()
                        dense_mask = np.isin(dense_mask, comps + 1)
                    else:
                        dense_mask = np.zeros_like(self.seg_data[:, :, section])

                dense_masks[section][structure] = dense_mask

        dense_masks = {section: reduce(np.add, [m for s, m in dense_cells.items()])
                       for section, dense_cells in dense_masks.items()}

        # heatmaps = dict()
        # for section in relevant_sections:
        #     heatmaps[section] = np.zeros_like(self.seg_data[:, :, section], dtype=float)
        #     heatmaps[section][csv[csv.section == section].centroid_y.to_numpy().astype(int) // 64,
        #                       csv[csv.section == section].centroid_x.to_numpy().astype(int) // 64] = \
        #         csv[csv.section == section].density
        #
        # import matplotlib.pyplot as plt
        # for section, mask in dense_masks.items():
        #     fig, axs = plt.subplots(1, 2)
        #     fig.suptitle(f"Section {section}")
        #     axs[0].imshow(mask, cmap='gray')
        #     axs[1].imshow(heatmaps.get(section, mask), cmap='hot')
        #     plt.show()

        for row in celldata_structs.itertuples():
            struct = dense_masks[row.section][int(row.centroid_y // 64), int(row.centroid_x) // 64]
            if struct != 0:
                csv.at[row.Index, 'pyramidal'] = True

    def process(self):
        self.logger.info(f"Extracting cell data for {self.id}...")
        sections = sorted([s for s in self.bboxes.keys() if self.bboxes[s]])
        if os.path.isfile(f'{self.directory}/celldata-{self.id}.csv') and os.path.isfile(f'{self.directory}/sectiondata-{self.id}.csv'):
            csv = pandas.read_csv(f'{self.directory}/celldata-{self.id}.csv')
            labels_to_remove = [c for c in csv if c.startswith('Unnamed')
                                or c.startswith('pyramidal') or c == 'density']
            csv = csv.drop(columns=labels_to_remove)
        else:
            section_data = list()
            cell_data = list()
            for section in sections:
                cells, sec = self.process_section(section)
                section_data.append(sec)
                cell_data += cells

            csv = pandas.DataFrame(section_data)
            csv.to_csv(f'{self.directory}/sectiondata-{self.id}.csv')
            csv = pandas.DataFrame(cell_data)

        self.logger.info(f"Calculating densities for {self.id}...")
        self.calculate_densities(csv)
        self.logger.info(f"Extracting pyramidal layers for {self.id}...")
        csv['pyramidal'] = False
        self.detect_pyramidal(csv)
        csv.to_csv(f'{self.directory}/celldata-{self.id}.csv')

        csv = pandas.DataFrame({f: self.details[f] for f in self.experiment_fields_to_save})
        csv.to_csv(f'{self.directory}/experimentdata-{self.id}.csv', index=False)

    def get_cell_mask(self, section, offset_x, offset_y, w, h, mask):
        cell_mask_file_name = os.path.join(self.directory,
                                           f'cellmask-{self.id}-{section}-{offset_x}_{offset_y}_{w}_{h}.png')
        cell_mask = cv2.imread(cell_mask_file_name, cv2.IMREAD_COLOR)[:, :, 1]
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        offset = np.array([offset_x, offset_y])
        cnts = [Polygon(cnt.squeeze() + offset) for cnt in cnts if cnt.shape[0] > 2]
        cnts = [poly for poly in cnts if mask[int(poly.centroid.y) // 64, int(poly.centroid.x) // 64]]
        return cnts

    def get_brain_area(self, section):
        thumbnail_file_name = os.path.join(self.directory, f'thumbnail-{self.id}-{section}.jpg')
        thumbnail = cv2.imread(thumbnail_file_name, cv2.IMREAD_GRAYSCALE)
        brain_mask, bbox, ctrs = detect_brain(thumbnail)
        brain_area = sum([Polygon(ctr.squeeze()).area for ctr in ctrs])
        return brain_area

    def process_section(self, section):
        brain_area = self.get_brain_area(section)
        struct_mask = self.get_structure_mask(section)

        self.logger.info(f"Experiment: {self.id}, processing section {section}...")
        struct_area = struct_mask.sum()

        cells_data = list()

        for offset_x, offset_y, w, h in map(lambda b: b.scale(64), self.bboxes[section]):
            cells = self.get_cell_mask(section, offset_x, offset_y, w, h, struct_mask)
            for cell in cells:
                struct_id = self.get_struct_id(cell, section)
                if struct_id in self.structure_ids:
                    cells_data.append({
                        'experiment': self.id,
                        'section': section,
                        'structure_id': struct_id,
                        'structure': self.structure_tree.get_name_map()[struct_id],
                        'centroid_x': cell.centroid.x,
                        'centroid_y': cell.centroid.y,
                        'area': cell.area * (self.subimages[section]['resolution'] ** 2),
                        'perimeter': cell.length * self.subimages[section]['resolution']
                    })

        return cells_data, {
            'experiment_id': self.id,
            'section_id': section,
            'brain_area': brain_area * ((self.subimages[section]['resolution'] * 64) ** 2),
            'struct_area': struct_area * ((self.subimages[section]['resolution'] * 64) ** 2)
        }

    def get_struct_id(self, cell, section):
        y = int(cell.centroid.y // 64)
        x = int(cell.centroid.x // 64)
        struct_id = self.seg_data[y, x, section]
        if struct_id == 0:
            neighborhood = self.seg_data[y - 1: y + 2, x - 1: x + 2, section]
            ids, counts = np.unique(neighborhood, return_counts=True)
            sorted_indices = np.argsort(-counts).tolist()
            for i in sorted_indices:
                if ids[i] != 0:
                    struct_id = ids[i]
                    break
            if struct_id == 0:
                struct_id = self.default_struct_id
        return struct_id


class CellProcessor(DirWatcher):
    experiment_fields_to_save = [
        'id',
        'gender',
        'injection_structures',
        'injection_volume',
        'injection_x',
        'injection_y',
        'injection_z',
        'product_id',
        'specimen_name',
        'strain',
        'structure_abbrev',
        'structure_id',
        'structure_name',
        'transgenic_line',
        'transgenic_line_id',
        'primary_injection_structure'
    ]

    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, structs, connectivity_dir,
                 _processor_number):
        super().__init__(input_dir, process_dir, output_dir, f'cell-processor-{_processor_number}')
        self.structure_ids = structs
        self.brain_seg_data_dir = structure_map_dir
        self.source_dir = input_dir
        self.output_dir = output_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.experiments = {int(e['id']): e for e in self.mcc.get_experiments(dataframe=False)}

    def process_item(self, item, directory):
        item = int(item)
        experiment = ExperimentCellsProcessor(self.mcc,
                                              item,
                                              directory,
                                              self.brain_seg_data_dir,
                                              self.structure_ids,
                                              self.experiment_fields_to_save,
                                              self.experiments[item],
                                              self.logger)
        experiment.process()


class ExperimentCellAnalyzerTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment cell data analyzer")

    def prepare_input(self, connectivity_dir, **kwargs):
        mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        mcc.get_structure_tree()

    def execute_task(self, structs, structure_map_dir, **kwargs):
        analyzer = CellProcessor(structs=ast.literal_eval(structs), structure_map_dir=structure_map_dir, **kwargs)
        experiments = os.listdir(structure_map_dir)
        analyzer.run_until_count(len(experiments))


if __name__ == '__main__':
    task_mgr = ExperimentCellAnalyzerTaskManager()
    task_mgr.run()
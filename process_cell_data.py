import argparse
import ast
import functools
import math
import operator as op
import os
import pickle
import time
import urllib.error
from collections import defaultdict

import cv2
import numpy as np
import pandas
import scipy.ndimage as ndi
import simplejson
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from annotate_cell_data import ExperimentDataAnnotator
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
        while True:
            try:
                self.details = {**details, **(mapi.get_experiment_detail(self.id)[0])}
                break
            except simplejson.errors.JSONDecodeError or urllib.error.URLError:
                time.sleep(1.0)
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
        structure_ids = list(set(functools.reduce(op.add, structure_ids)))
        return structure_ids

    def get_structure_mask(self, section):
        section_seg_data = self.seg_data[:, :, section]
        mask = np.isin(section_seg_data, self.structure_ids)
        mask = ndi.binary_closing(ndi.binary_fill_holes(mask).astype(np.int8)).astype(np.int8)
        return mask

    def calculate_coverages(self, csv):
        csv['coverage'] = 0
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
            coverages = np.array([image[centroids_y[i] - 32: centroids_y[i] + 32,
                                  centroids_x[i] - 32: centroids_x[i] + 32].sum() for i in range(len(centroids_y))])
            csv.at[csv.section == section, 'coverage'] = coverages / 4096

    def build_dense_masks(self, celldata_struct, dense_masks, relevant_sections):
        scale = 64 // (dense_masks.shape[0] // self.seg_data.shape[0])
        for section in relevant_sections:
            celldata_section = celldata_struct[celldata_struct.section == section]
            centroids_x = (celldata_section.centroid_x.to_numpy() // scale).astype(int)
            centroids_y = (celldata_section.centroid_y.to_numpy() // scale).astype(int)
            coverages = celldata_section.coverage.to_numpy()

            if coverages.shape[0] > 2:
                model = KMeans(n_clusters=2)
                yhat = model.fit_predict(coverages.reshape(-1, 1))
                clusters = np.unique(yhat).tolist()
                if len(clusters) == 1:
                    yhat = np.zeros_like(coverages)
                    dense = 1
                else:
                    dense = np.argmax([coverages[yhat == 0].mean(), coverages[yhat == 1].mean()])
                if scale < 8:
                    dense_masks[:, :, section - min(relevant_sections)] = self.produce_precise_dense_mask(
                        celldata_section, centroids_x, centroids_y, dense, dense_masks, scale, yhat)
                else:
                    dense_masks[:, :, section - min(relevant_sections)] = \
                        self.produce_coarse_dense_mask(centroids_x, centroids_y, dense, yhat)

    def produce_precise_dense_mask(self, celldata_section, centroids_x, centroids_y, dense, dense_masks, scale, yhat):
        dense_mask = np.zeros_like(dense_masks[:, :, 0], dtype=np.uint8)
        radius = (math.sqrt(celldata_section.area.max() / math.pi) + 0.5) * 8 / scale
        centroids_y, centroids_x = centroids_y[yhat == dense], centroids_x[yhat == dense]
        radii = np.maximum((celldata_section.coverage.to_numpy() * radius + 0.5), 1).astype(int)
        for i in range(centroids_x.shape[0]):
            cv2.circle(dense_mask, (centroids_x[i], centroids_y[i]), radii[i], 1, cv2.FILLED)
        dense_mask = ndi.binary_dilation(dense_mask, ndi.generate_binary_structure(2, 64 // scale), iterations=6)
        return self.remove_small_components(dense_mask)

    def produce_coarse_dense_mask(self, centroids_x, centroids_y, dense, yhat):
        dense_mask = np.zeros_like(self.seg_data[:, :, 0])
        dense_mask[centroids_y[yhat == dense], centroids_x[yhat == dense]] = 1
        dense_mask = ndi.binary_closing(dense_mask, ndi.generate_binary_structure(2, 1), iterations=4)
        return self.remove_small_components(dense_mask)

    def remove_small_components(self, dense_mask):
        dense_mask, comps = ndi.measurements.label(dense_mask)
        if comps > 0:
            dm_nonzero = dense_mask[dense_mask != 0]
            sums = np.array([(dm_nonzero == i + 1).sum() for i in range(comps)])
            comps = np.argwhere((sums > 10) & ((sums.max() / sums) < 5)).flatten()
            dense_mask = np.isin(dense_mask, comps + 1)
        else:
            dense_mask = np.zeros_like(self.seg_data[:, :, 0])
        return dense_mask

    def plot_coverage_masks(self, data_frame, dense_masks, relevant_sections):
        heatmaps = dict()
        for section in relevant_sections:
            heatmaps[section] = np.zeros_like(self.seg_data[:, :, section], dtype=float)
            heatmaps[section][data_frame[data_frame.section == section].centroid_y.to_numpy().astype(int) // 64,
                              data_frame[data_frame.section == section].centroid_x.to_numpy().astype(int) // 64] = \
                data_frame[data_frame.section == section].coverage
        import matplotlib.pyplot as plt
        for section in range(dense_masks.shape[2]):
            fig, axs = plt.subplots(1, 2)
            fig.suptitle(f"Section {section + min(relevant_sections)}")
            axs[0].imshow(dense_masks[:, :, section], cmap='gray')
            axs[1].imshow(heatmaps.get(section + min(relevant_sections), np.zeros_like(dense_masks[:, :, section])),
                          cmap='hot')
            plt.show()

    def detect_dense_dg(self, data_frame):
        dg_structs = [10703, 10704, 632]
        scale = 4
        celldata_struct = data_frame[data_frame.structure_id.isin(dg_structs)]
        relevant_sections = sorted(np.unique(celldata_struct.section.to_numpy()).tolist())
        if not relevant_sections:
            return {'dense': 0, 'sparse': 0}

        dense_masks = np.zeros((self.seg_data.shape[0] * 64 // scale, self.seg_data.shape[1] * 64 // scale,
                                max(relevant_sections) - min(relevant_sections) + 1), dtype=np.uint8)

        self.build_dense_masks(celldata_struct, dense_masks, relevant_sections)
        # self.plot_coverage_masks(data_frame, dense_masks, relevant_sections)

        centroids_y = celldata_struct.centroid_y.to_numpy().astype(int) // scale
        centroids_x = celldata_struct.centroid_x.to_numpy().astype(int) // scale
        sections = celldata_struct.section.to_numpy().astype(int)
        data_frame.at[data_frame.structure_id.isin(dg_structs), 'dense'] = \
            dense_masks[centroids_y, centroids_x, sections - min(relevant_sections)].astype(bool)

        data_frame.at[data_frame.structure_id.isin([10703, 10704]) & data_frame.dense, 'structure_id'] = 632

        sparse = data_frame[(data_frame.structure_id == 632) & (data_frame.dense == False)]
        sparse_neighbours = data_frame[data_frame.structure_id.isin([10703, 10704])]

        x = np.stack((sparse_neighbours.centroid_x.to_numpy(), sparse_neighbours.centroid_y.to_numpy())).swapaxes(0, 1)
        y = sparse_neighbours.structure_id.to_numpy()

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(x, y)

        x = np.stack((sparse.centroid_x.to_numpy(), sparse.centroid_y.to_numpy())).swapaxes(0, 1)
        y = neigh.predict(x).tolist()

        data_frame.at[(data_frame.structure_id == 632) & (data_frame.dense == False), 'structure_id'] = y

        dense_dg_area = np.sum([dense_masks[:, :, i].sum() * (self.subimages[min(relevant_sections) + i]
                                                                ['resolution'] * 4) ** 2 for i in
                             range(dense_masks.shape[2])])
        dg_area = np.sum(
            [np.isin(self.seg_data[:, :, i], [10703, 10704, 632]).sum() * (self.subimages[i]['resolution'] * 64) ** 2
             for i in relevant_sections])

        return {'dense': dense_dg_area, 'sparse': dg_area - dense_dg_area}

    def detect_dense_ca(self, csv):
        structures_including_dense = [f'Field CA{i}' for i in range(1, 4)]
        structures_including_dense = [r['id'] for r in
                                      self.structure_tree.get_structures_by_name(structures_including_dense)]
        celldata_structs = csv[csv.structure_id.isin(structures_including_dense)]
        relevant_sections = sorted(np.unique(celldata_structs.section.to_numpy()).tolist())
        if not relevant_sections:
            return {403: {'dense': 0, 'sparse': 0}}

        dense_masks = np.zeros((self.seg_data.shape[0], self.seg_data.shape[1],
                                max(relevant_sections) - min(relevant_sections) + 1,
                                len(structures_including_dense)), dtype=int)

        areas = defaultdict(dict)

        for ofs, structure in enumerate(structures_including_dense):
            celldata_struct = celldata_structs[celldata_structs.structure_id.isin([structure])]
            self.build_dense_masks(celldata_struct, dense_masks[:, :, :, ofs], relevant_sections)
            dense_area = np.sum([dense_masks[:, :, i, ofs].sum() * (self.subimages[min(relevant_sections) + i]
                                                                    ['resolution'] * 64) ** 2 for i in
                                 range(dense_masks.shape[2])])
            total_area = np.sum(
                [(self.seg_data[:, :, i] == structure).sum() * (self.subimages[i]['resolution'] * 64) ** 2
                 for i in relevant_sections])
            areas[structure] = {'dense': dense_area, 'sparse': total_area - dense_area}

        dense_masks = dense_masks.sum(axis=3) != 0
        # self.plot_coverage_masks(csv, dense_masks, relevant_sections)

        centroids_y = celldata_structs.centroid_y.to_numpy().astype(int) // 64
        centroids_x = celldata_structs.centroid_x.to_numpy().astype(int) // 64
        sections = celldata_structs.section.to_numpy().astype(int)
        csv.at[csv.structure_id.isin(structures_including_dense), 'dense'] = \
            dense_masks[centroids_y, centroids_x, sections - min(relevant_sections)].astype(bool)

        return areas

    def process(self):
        self.logger.info(f"Extracting cell data for {self.id}...")
        sections = sorted([s for s in self.bboxes.keys() if self.bboxes[s]])
        section_data = list()
        cell_data = list()
        for section in sections:
            cells, sec = self.process_section(section)
            section_data.append(sec)
            cell_data += cells

        cell_dataframe = pandas.DataFrame(section_data)
        cell_dataframe.to_csv(f'{self.directory}/sectiondata-{self.id}.csv')
        cell_dataframe = pandas.DataFrame(cell_data)
        self.logger.info(f"Calculating coverages for {self.id}...")
        self.calculate_coverages(cell_dataframe)
        self.logger.info(f"Extracting dense layers for CA regions in {self.id}...")
        cell_dataframe['dense'] = False
        dense_areas = self.detect_dense_ca(cell_dataframe)

        self.logger.info(f"Extracting dense layers for DG regions in {self.id}...")
        dense_areas[632] = self.detect_dense_dg(cell_dataframe)
        self.logger.info(f"Saving cell data for {self.id}...")
        cell_dataframe.to_parquet(f'{self.directory}/celldata-{self.id}.parquet')

        cell_dataframe = pandas.DataFrame({f: self.details[f] for f in self.experiment_fields_to_save})
        cell_dataframe.to_csv(f'{self.directory}/experimentdata-{self.id}.csv', index=False)
        with open(f'{self.directory}/areas.pickle', 'wb') as f:
            pickle.dump(dense_areas, file=f)

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

        self.logger.debug(f"Experiment: {self.id}, processing section {section}...")
        struct_area = struct_mask.sum()

        cells_data = list()

        for offset_x, offset_y, w, h in map(lambda b: b.scale(64), self.bboxes[section]):
            cells = self.get_cell_mask(section, offset_x, offset_y, w, h, struct_mask)
            box_cell_data = self.polygons_to_cell_data(cells, section)
            cells_data += box_cell_data

        return cells_data, {
            'experiment_id': self.id,
            'section_id': section,
            'brain_area': brain_area * ((self.subimages[section]['resolution'] * 64) ** 2),
            'struct_area': struct_area * ((self.subimages[section]['resolution'] * 64) ** 2)
        }

    def polygons_to_cell_data(self, cells, section):
        struct_ids = [self.get_struct_id(cell, section) for cell in cells]
        box_cell_data = [{'section': section, 'structure_id': struct_id, 'centroid_x': int(cell.centroid.x),
                          'centroid_y': int(cell.centroid.y),
                          'area': cell.area * (self.subimages[section]['resolution'] ** 2),
                          'perimeter': cell.length * self.subimages[section]['resolution'], } for cell, struct_id in
                         zip(cells, struct_ids) if struct_id in self.structure_ids]
        return box_cell_data

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

    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, structs, connectivity_dir, annotate,
                 _processor_number):
        super().__init__(input_dir, process_dir, output_dir, f'cell-processor-{_processor_number}')
        self.annotate = annotate
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

        if self.annotate:
            annotator = ExperimentDataAnnotator(int(item), directory, self.logger)
            annotator.process()


class ExperimentCellAnalyzerTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment cell data analyzer")

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument('--annotate', action='store_true', default=False,
                            help='Annotate after processing')

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

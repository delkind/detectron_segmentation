import argparse
import ast
import functools
import operator as op
import os
import pickle
from collections import defaultdict
from functools import reduce

import numpy as np
import cv2
import pandas
from shapely.geometry import Polygon
import scipy.ndimage as ndi

from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from sklearn.cluster import KMeans

from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager
from localize_brain import detect_brain


class ExperimentPyramidalExtractor(object):
    def __init__(self, experiment_id, directory, structdata_dir, celldata, method, mcc, column_name, logger):
        self.column_name = column_name
        self.method = method
        self.logger = logger
        self.structdata_dir = structdata_dir
        self.directory = directory
        self.experiment_id = experiment_id
        tree = mcc.get_structure_tree()
        self.structures_including_pyramidal = [f'Field CA{i}' for i in range(1, 4)]
        self.pyramidal_layers = {s: self.get_pyramidal_layer(tree, s) for s in self.structures_including_pyramidal}
        self.celldata = celldata

    def create_heatmap(self, celldata_struct, section, binsize=1):
        thumb = cv2.imread(f"{self.directory}/thumbnail-{self.experiment_id}-{section}.jpg",
                           cv2.IMREAD_GRAYSCALE)
        section_celldata = celldata_struct.loc[celldata_struct.section == section]
        heatmap = np.zeros((thumb.shape[0] // binsize, thumb.shape[1] // binsize), dtype=int)
        x = (section_celldata.centroid_x.to_numpy() // 64 // binsize).astype(int)
        y = (section_celldata.centroid_y.to_numpy() // 64 // binsize).astype(int)
        densities = (section_celldata.density.to_numpy()).astype(int)
        for i in range(len(section_celldata)):
            heatmap[y[i], x[i]] = self.method(heatmap[y[i], x[i]], densities[i])
        heatmap = np.kron(heatmap, np.ones((binsize, binsize)))
        return heatmap, thumb

    @staticmethod
    def process_section(heatmap, mask):
        coords = np.where(heatmap != 0)
        cells = heatmap[coords]
        dense_cells = np.zeros_like(heatmap)
        if cells.shape[0] > 2:
            model = KMeans(n_clusters=2)
            yhat = model.fit_predict(cells.reshape(-1, 1) ** 2)
            clusters = np.unique(yhat).tolist()
            if len(clusters) == 1:
                yhat = np.zeros_like(cells)
                dense = 1
            else:
                dense = np.argmax([cells[yhat == 0].mean(), cells[yhat == 1].mean()])

            dense_cells[coords[0][(yhat == dense)], coords[1][(yhat == dense)]] = 1
            dense_cells = ndi.binary_dilation(dense_cells, ndi.generate_binary_structure(2, 1), iterations=2)
            dense_cells = ndi.binary_closing(dense_cells, ndi.generate_binary_structure(2, 1), iterations=2)
            dense_cells, comps = ndi.measurements.label(dense_cells)
            if comps > 0:
                sums = np.array([(dense_cells == i + 1).sum() for i in range(comps)])
                comps = np.argwhere((sums.max() / sums) < 5).flatten()
                dense_cells = np.logical_and(np.isin(dense_cells, comps + 1), mask)

        retval = np.zeros_like(dense_cells, dtype=int)
        retval[dense_cells.astype(bool)] = 1

        return retval

    @staticmethod
    def get_pyramidal_layer(tree, structure_name):
        descendants = [tree.get_structures_by_id(i) for i in tree.descendant_ids([s['id'] for s in
                                                                                  tree.get_structures_by_name(
                                                                                      [structure_name])])]
        descendants = [(i['name'], i['id']) for s in descendants for i in s if i['name'].find('pyramidal') > -1]
        return descendants[0]

    def process(self):
        structure_data = np.load(f'{self.structdata_dir}/{self.experiment_id}/'
                                 f'{self.experiment_id}-sections.npz')['arr_0']

        dense_masks = defaultdict(dict)
        heatmaps = defaultdict(dict)
        celldata_structs = self.celldata[self.celldata.structure.isin(self.structures_including_pyramidal)]
        relevant_sections = sorted(np.unique(celldata_structs.section.to_numpy()).tolist())

        for structure in self.structures_including_pyramidal:
            celldata_struct = celldata_structs.loc[self.celldata.structure == structure]
            struct_id = np.unique(celldata_struct.structure_id.to_numpy())
            mask = structure_data == struct_id

            heatmaps[structure] = {s: self.create_heatmap(celldata_struct, s, 2) for s in relevant_sections}

            for section, data in heatmaps[structure].items():
                heatmap, _ = data
                dense_masks[section][structure] = self.process_section(heatmap, mask[:, :, section])

        dense_masks = {section: reduce(np.add, [m * self.pyramidal_layers[s][1] for s, m in dense_cells.items()])
                       for section, dense_cells in dense_masks.items()}

        for row in celldata_structs.itertuples():
            struct = dense_masks[row.section][int(row.centroid_y // 64), int(row.centroid_x) // 64]
            if struct != 0:
                self.celldata.at[row.Index, self.column_name] = True


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

    def get_structure_children(self):
        structure_ids = self.structure_tree.descendant_ids(self.parent_struct_id)
        structure_ids = set(functools.reduce(op.add, structure_ids))
        return structure_ids

    def get_structure_mask(self, section):
        section_seg_data = self.seg_data[:, :, section]
        mask = np.zeros_like(section_seg_data, dtype=np.int8)
        for stid in self.structure_ids:
            mask[section_seg_data == stid] = True
        mask = ndi.binary_closing(ndi.binary_fill_holes(mask).astype(np.int8)).astype(np.int8)
        return mask

    @staticmethod
    def calculate_densities(centroids, radius=84):
        densities = np.zeros(centroids.shape[0], dtype=np.int16)
        for i in range(centroids.shape[0]):
            dists = np.max(np.abs(centroids - centroids[i, :]), axis=1) < radius
            densities[i] = dists.sum()
        return densities

    def process(self):
        with open(f'{self.directory}/bboxes.pickle', 'rb') as f:
            bboxes = pickle.load(f)
        sections = sorted([s for s in bboxes.keys() if bboxes[s]])

        if os.path.isfile(f'{self.directory}/celldata-{self.id}.csv') and os.path.isfile(f'{self.directory}/sectiondata-{self.id}.csv'):
            csv = pandas.read_csv(f'{self.directory}/celldata-{self.id}.csv')
        else:
            section_data = list()
            cell_data = list()
            for section in sections:
                cells, sec = self.process_section(section, bboxes[section])
                section_data.append(sec)
                cell_data += cells

            csv = pandas.DataFrame(section_data)
            csv.to_csv(f'{self.directory}/sectiondata-{self.id}.csv')
            csv = pandas.DataFrame(cell_data)
            centroids = np.array([csv.centroid_x, csv.centroid_y]).swapaxes(0, 1).astype(np.int16)
            densities = self.calculate_densities(centroids)
            csv['density'] = np.array(densities)

        csv['pyramidal'] = False
        csv['pyramidal_grid'] = False

        self.logger.info(f"Extracting pyramidal layers for {self.id}...")
        extractor = ExperimentPyramidalExtractor(self.id, self.directory, self.brain_seg_data_dir, csv,
                                                 lambda c, d: max(c, d), self.mcc, 'pyramidal', self.logger)
        extractor.process()
        self.logger.info(f"Extracting grid-based pyramidal layers for {self.id}...")
        extractor = ExperimentPyramidalExtractor(self.id, self.directory, self.brain_seg_data_dir, csv,
                                                 lambda c, d: c + 1, self.mcc, 'pyramidal_grid', self.logger)
        extractor.process()
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

    def process_section(self, section, bboxes):
        brain_area = self.get_brain_area(section)
        struct_mask = self.get_structure_mask(section)

        self.logger.info(f"Experiment: {self.id}, processing section {section}...")
        struct_area = struct_mask.sum()

        cells_data = list()

        for offset_x, offset_y, w, h in map(lambda b: b.scale(64), bboxes):
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
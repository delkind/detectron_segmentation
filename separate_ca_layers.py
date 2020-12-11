import argparse
import os
from collections import defaultdict
from functools import reduce

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from sklearn.cluster import KMeans

from dir_watcher import DirWatcher
from task_manager import TaskManager


class ExperimentPyramidalExtractor(object):
    def __init__(self, experiment_id, directory, structdata_dir, connectivity_dir, logger):
        self.logger = logger
        self.structdata_dir = structdata_dir
        self.directory = directory
        self.experiment_id = experiment_id
        mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        tree = mcc.get_structure_tree()
        self.structures = [f'Field CA{i}' for i in range(1, 4)]
        self.pyramidal_layers = {s: self.get_pyramidal_layer(tree, s) for s in self.structures}

    def create_heatmap(self, celldata, celldata_struct, section, binsize=1):
        thumb = cv2.imread(f"{self.directory}/thumbnail-{self.experiment_id}-{section}.jpg",
                           cv2.IMREAD_GRAYSCALE)
        section_celldata = celldata_struct.loc[celldata.section == section]
        heatmap = np.zeros((thumb.shape[0] // binsize, thumb.shape[1] // binsize), dtype=int)
        x = (section_celldata.centroid_x.to_numpy() // 64 // binsize).astype(int)
        y = (section_celldata.centroid_y.to_numpy() // 64 // binsize).astype(int)
        for i in range(len(section_celldata)):
            heatmap[y[i], x[i]] += 1
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
        self.logger.info(f"Processing {self.experiment_id}...")
        celldata = pd.read_csv(f'{self.directory}/celldata-{self.experiment_id}.csv')
        structure_data = np.load(f'{self.structdata_dir}/{self.experiment_id}/'
                                 f'{self.experiment_id}-sections.npz')['arr_0']

        dense_masks = defaultdict(dict)
        heatmaps = defaultdict(dict)
        celldata_structs = celldata[celldata.structure.isin(self.structures)]
        relevant_sections = sorted(np.unique(celldata_structs.section.to_numpy()).tolist())

        for structure in self.structures:
            celldata_struct = celldata_structs.loc[(celldata.structure == structure)]
            struct_id = np.unique(celldata_struct.structure_id.to_numpy())
            mask = structure_data == struct_id

            heatmaps[structure] = {s: self.create_heatmap(celldata, celldata_struct, s, 2) for s in relevant_sections}

            for section, data in heatmaps[structure].items():
                heatmap, _ = data
                dense_masks[section][structure] = self.process_section(heatmap, mask[:, :, section])

        dense_masks = {section: reduce(np.add, [m * self.pyramidal_layers[s][1] for s, m in dense_cells.items()])
                       for section, dense_cells in dense_masks.items()}

        pyramidal_layers = {id: name for name, id in self.pyramidal_layers.values()}

        # celldata_struct = celldata.loc[celldata.structure.isin(self.structures)]
        # heatmaps = {s: self.create_heatmap(celldata, celldata_struct, s, 2) for s in
        #             sorted(np.unique(celldata_struct.section.to_numpy()).tolist())}
        #
        # for section, dense_cells in dense_masks.items():
        #     heatmap, _ = heatmaps[section]
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(dense_cells != 0, cmap='gray')
        #     ax[1].imshow(heatmap, cmap='hot')
        #     fig.suptitle(f"Section {section}")
        #     plt.show()

        for row in celldata_structs.itertuples():
            struct = dense_masks[row.section][int(row.centroid_y // 64), int(row.centroid_x) // 64]
            if struct != 0:
                celldata.at[row.Index, 'structure_id'] = struct
                celldata.at[row.Index, 'structure'] = pyramidal_layers[struct]

        celldata.to_csv(f'{self.directory}/pyr_celldata-{self.experiment_id}.csv')


class CellProcessor(DirWatcher):
    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, connectivity_dir,
                 _processor_number):
        super().__init__(input_dir, process_dir, output_dir, f'pyramidal-extractor-{_processor_number}')
        self.connectivity_dir = connectivity_dir
        self.brain_seg_data_dir = structure_map_dir
        self.source_dir = input_dir
        self.output_dir = output_dir

    def process_item(self, item, directory):
        experiment = ExperimentPyramidalExtractor(item, directory, structdata_dir=self.brain_seg_data_dir,
                                                  connectivity_dir=self.connectivity_dir, logger=self.logger)
        experiment.process()


class ExperimentCellAnalyzerTaskManager(TaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment cell pyramidal layer extractor")

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument('--input-dir', '-i', action='store', required=True, help='Input directory')
        parser.add_argument('--process_dir', '-d', action='store', required=True, help='Processing directory')
        parser.add_argument('--output_dir', '-o', action='store', required=True,
                            help='Results output directory')
        parser.add_argument('--connectivity_dir', '-c', action='store', required=True,
                            help='Connectivity cache directory')
        parser.add_argument('--structure_map_dir', '-m', action='store', required=True,
                            help='Brain structure map directory')

    def prepare_input(self, connectivity_dir, **kwargs):
        pass

    def execute_task(self, structure_map_dir, **kwargs):
        analyzer = CellProcessor(structure_map_dir=structure_map_dir, **kwargs)
        experiments = os.listdir(structure_map_dir)
        analyzer.run_until_count(len(experiments))


if __name__ == '__main__':
    task_mgr = ExperimentCellAnalyzerTaskManager()
    task_mgr.run()

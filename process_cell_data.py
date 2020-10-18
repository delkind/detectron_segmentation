import argparse
import functools
import operator as op
import os
import re

import numpy as np
import cv2
import pandas
from shapely.geometry import Polygon
import scipy.ndimage as ndi

from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from build_cell_data import create_cell_build_argparser
from dir_watcher import DirWatcher
from localize_brain import detect_brain


class ExperimentCellsProcessor(object):
    def __init__(self, mcc, experiment_id, input_dir, cache_dir, output_dir, brain_seg_data_dir, parent_struct_id,
                 experiment_fields_to_save, details, verify_thumbnail, logger, default_struct_id=997):
        self.verify_thumbnail = verify_thumbnail
        self.experiment_fields_to_save = experiment_fields_to_save
        self.default_struct_id = default_struct_id
        self.parent_struct_id = parent_struct_id
        self.brain_seg_data_dir = brain_seg_data_dir
        self.cache_dir = cache_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.mcc = mcc
        self.id = experiment_id
        mapi = MouseConnectivityApi()
        self.details = {**details, **(mapi.get_experiment_detail(self.id)[0])}
        self.logger = logger
        self.subimages = {i['section_number']: i for i in self.details['sub_images']}
        self.seg_data = np.load(f'{self.brain_seg_data_dir}/{self.id}-sections.npz')['arr_0']
        self.structure_tree = self.mcc.get_structure_tree()
        self.structure_ids = self.get_structure_children()

    def load_hippo_mask(self, section):
        hippo_mask_file_name = os.path.join(self.input_dir, f'{self.id}-{section}-hippomask.png')
        hippo_mask = cv2.imread(hippo_mask_file_name, cv2.IMREAD_GRAYSCALE)
        if hippo_mask is not None:
            hippo_ctrs, _ = cv2.findContours(hippo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hippo_mask = cv2.fillPoly(hippo_mask, hippo_ctrs, color=255).astype(bool)
        return hippo_mask

    def get_structure_children(self):
        structure_ids = self.structure_tree.descendant_ids([self.parent_struct_id])
        structure_ids = set(functools.reduce(op.add, structure_ids))
        return structure_ids

    def get_structure_mask(self, section):
        section_seg_data = self.seg_data[:, :, section]
        mask = np.zeros_like(section_seg_data, dtype=np.int8)
        for stid in self.structure_ids:
            mask[section_seg_data == stid] = True
        mask = ndi.binary_closing(ndi.binary_fill_holes(mask).astype(np.int8)).astype(np.int8)
        return mask

    def process(self):
        full_sections = sorted([tuple([int(s) for s in re.findall('\\d+', f)]) for f in os.listdir(self.cache_dir)
                                if f.endswith('-full.jpg')], key=lambda t: t[-1])

        section_data = list()
        for offset_x, offset_y, _, _, section in full_sections:
            result = self.process_section(section, offset_x, offset_y)
            if result is not None:
                section_data.append(result)

        csv = pandas.DataFrame(section_data)
        csv.to_csv(f'{self.output_dir}/{self.id}-sectiondata.csv')
        return {f: self.details[f] for f in self.experiment_fields_to_save}

    def get_cell_mask(self, section, offset_x, offset_y):
        cell_mask_file_name = os.path.join(self.input_dir, f'{self.id}-{section}-cellmask.png')
        cell_mask = cv2.imread(cell_mask_file_name, cv2.IMREAD_GRAYSCALE)
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [Polygon(cnt.squeeze()) for cnt in cnts if cnt.shape[0] > 2]
        mask = self.get_structure_mask(section)
        cnts = [poly for poly in cnts if mask[int(poly.centroid.y + offset_y) // 64,
                                              int(poly.centroid.x + offset_x) // 64]]
        return cnts

    def get_brain_area(self, section):
        thumbnail_file_name = os.path.join(self.input_dir, f'{self.id}-{section}-thumb.jpg')
        thumbnail = cv2.imread(thumbnail_file_name, cv2.IMREAD_GRAYSCALE)
        brain_mask, bbox, ctrs = detect_brain(thumbnail)
        brain_area = sum([Polygon(ctr.squeeze()).area for ctr in ctrs]) * 64 * 64
        return brain_area

    def process_section(self, section, offset_x, offset_y):
        brain_area = self.get_brain_area(section)
        struct_mask = self.get_structure_mask(section)

        if not self.is_acceptable_iou(offset_x, offset_y, section, struct_mask):
            self.logger.info(f"Experiment: {self.id}, skipping section {section}, partial structure")
            return None

        self.logger.info(f"Experiment: {self.id}, processing section {section}...")
        struct_area = struct_mask.sum()
        cells = self.get_cell_mask(section, offset_x, offset_y)

        cells_data = list()

        for cell in cells:
            struct_id = self.get_struct_id(cell, offset_x, offset_y, section)
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

        csv = pandas.DataFrame(cells_data)
        csv.to_csv(f'{self.output_dir}/{self.id}-{section}-celldata.csv')

        return {
            'experiment_id': self.id,
            'section_id': section,
            'brain_area': brain_area * ((self.subimages[section]['resolution'] * 64) ** 2),
            'struct_area': struct_area * ((self.subimages[section]['resolution'] * 64) ** 2)
        }

    def is_acceptable_iou(self, offset_x, offset_y, section, struct_mask):
        if not self.verify_thumbnail:
            return True
        hippo_mask = self.load_hippo_mask(section)
        if hippo_mask is None:
            return True
        hippo_mask = cv2.resize(hippo_mask.astype(np.uint8), (0, 0), fx=1 / 64, fy=1 / 64)
        hippo_msk = np.zeros_like(struct_mask)
        hippo_msk[offset_y // 64: offset_y // 64 + hippo_mask.shape[0],
        offset_x // 64: offset_x // 64 + hippo_mask.shape[1]] = hippo_mask
        intersection = (struct_mask & hippo_msk).sum()
        union = (struct_mask | hippo_msk).sum()
        iou = intersection / union
        return iou > 0.8

    def get_struct_id(self, cell, offset_x, offset_y, section):
        y = int((cell.centroid.y + offset_y) // 64)
        x = int((cell.centroid.x + offset_x) // 64)
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

    def __init__(self, input_dir, output_dir, brain_seg_data_dir, structure_id, verify_thumbnail, number):
        super().__init__(*[os.path.join(output_dir, d) for d in ['input', 'proc', 'result']],
                         f'cell-processor-{number}')
        self.verify_thumbnail = verify_thumbnail
        self.structure_id = structure_id
        self.brain_seg_data_dir = brain_seg_data_dir
        self.source_dir = input_dir
        self.output_dir = output_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'{self.source_dir}'
                                                        f'/connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.experiments = {int(e['id']): e for e in self.mcc.get_experiments(dataframe=False)}

    def process_item(self, item, directory):
        item = int(item)
        experiment = ExperimentCellsProcessor(self.mcc, item, f'{self.source_dir}/{item}/',
                                              f'{self.source_dir}/cache/{item}/',
                                              directory,
                                              f'{self.brain_seg_data_dir}/{item}',
                                              self.structure_id,
                                              self.experiment_fields_to_save,
                                              self.experiments[item],
                                              self.verify_thumbnail,
                                              self.logger)
        return experiment.process()

    def reduce(self, results, output_dir):
        csv = pandas.DataFrame(results)
        csv.to_csv(f'{output_dir}/expdata.csv')


if __name__ == '__main__':
    parser = create_cell_build_argparser()
    parser.add_argument('--number', '-n', action='store', type=int, required=True, help='Number of this instance')
    args = parser.parse_args()

    print(vars(args))
    processor = CellProcessor(**vars(args))
    processor.run_until_empty()

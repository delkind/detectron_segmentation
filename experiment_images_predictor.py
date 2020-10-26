import argparse
import ast
import os
import pickle
import re
from collections import defaultdict

import cv2
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager
from predict_experiment import create_crops_list, extract_predictions


class ExperimentImagesPredictor(DirWatcher):
    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, parent_structs, connectivity_dir, number,
                 cell_model, crop_size, border_size, device, threshold):
        super().__init__(input_dir, process_dir, output_dir, f'experiment-images-predictor-{number}')
        self.threshold = threshold
        self.device = device
        self.border_size = border_size
        self.crop_size = crop_size
        self.cell_model = cell_model
        self.parent_structs = parent_structs
        self.segmentation_dir = structure_map_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        struct_tree = self.mcc.get_structure_tree()
        structure_ids = [i for sublist in struct_tree.descendant_ids(self.parent_structs) for i in sublist]
        self.structure_ids = set(structure_ids)
        self.bbox_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
        self.init_model()

    def init_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.MODEL.WEIGHTS = self.cell_model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cell_model = DefaultPredictor(cfg)

    def process_item(self, item, directory):
        experiment_id = int(item)
        segmentation = np.load(f'{self.segmentation_dir}/{item}/{item}-sections.npz')['arr_0']
        mask = np.isin(segmentation, list(self.structure_ids))
        with open(f'{directory}/bboxes.pickle', 'rb') as f:
            bboxes = pickle.load(f)
        sections = sorted([s for s in bboxes.keys() if bboxes[s]])
        for section in sorted(sections):
            self.process_section(directory, experiment_id, section, bboxes[section], mask[:, :, section])

    def process_section(self, directory, experiment_id, section, bboxes, mask):
        self.logger.info(f"Experiment {experiment_id}: processing section {section}...")
        for bbox in bboxes:
            x, y, w, h = bbox.scale(64)
            image = cv2.imread(f'{directory}/full-{experiment_id}-{section}-{x}_{y}_{w}_{h}.jpg',
                               cv2.IMREAD_GRAYSCALE)
            img_mask = self.predict_cells(image, mask, x, y)
            cv2.imwrite(f'{directory}/cellmask-{experiment_id}-{section}-{x}_{y}_{w}_{h}.png',
                        np.stack([np.zeros_like(img_mask), img_mask, np.zeros_like(img_mask)], axis=2))

    def predict_cells(self, image, section_mask, x, y):
        crops = create_crops_list(self.border_size, self.crop_size, image)
        cell_mask = np.zeros_like(image)
        mask_basex, mask_basey, mask_crop_size = map(lambda v: v // 64, [x, y, self.crop_size])
        for crop, coords in crops:
            ym, xm = coords[0] // 64 + mask_basey, coords[1] // 64 + mask_basex
            if section_mask[ym: min(ym + mask_crop_size + 1, section_mask.shape[0] - 1),
               xm: min(xm + mask_crop_size + 1, section_mask.shape[1] - 1)].sum() > 0:
                outputs = self.cell_model(cv2.cvtColor(image[coords[0]: coords[0] + self.crop_size,
                                          coords[1]: coords[1] + self.crop_size], cv2.COLOR_GRAY2BGR))
                _, mask = extract_predictions(outputs["instances"].to("cpu"))
                if mask is not None:
                    cell_mask[coords[0]: coords[0] + self.crop_size, coords[1]: coords[1] + self.crop_size] = \
                        np.logical_or(cell_mask[coords[0]: coords[0] + self.crop_size,
                                      coords[1]: coords[1] + self.crop_size], mask)

        ctrs, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(cell_mask)
        cv2.fillPoly(mask, ctrs, color=255)
        ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(cell_mask)
        cv2.polylines(mask, ctrs, isClosed=True, color=255)
        return mask


class ExperimentDownloadTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment downloader")

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument('--cell_model', action='store', required=True, help='Cell segmentation model')
        parser.add_argument('--crop_size', default=312, type=int, action='store', help='Size of a single crop')
        parser.add_argument('--border_size', default=20, type=int, action='store',
                            help='Size of the border (to make crops overlap)')
        parser.add_argument('--device', default='cuda', action='store', help='Model execution device')
        parser.add_argument('--threshold', default=0.5, action='store', help='Prediction threshold')

    def prepare_input(self, connectivity_dir, **kwargs):
        mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        mcc.get_structure_tree()

    def execute_task(self, structs, **kwargs):
        downloader = ExperimentImagesPredictor(parent_structs=ast.literal_eval(structs), **kwargs)
        downloader.run_until_empty()


if __name__ == '__main__':
    dl = ExperimentDownloadTaskManager()
    dl.run()

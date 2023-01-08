import argparse
import ast
import itertools
import os
import pickle

import cv2
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import GenericMask
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager
from shapely.geometry import Polygon


def extract_predictions(predictions):
    if not predictions.has('pred_masks'):
        return None, None

    masks = np.asarray(predictions.pred_masks)
    if masks.shape[0] == 0:
        return None, None

    mask = np.zeros_like(masks[0, :, :])
    for m in masks:
        mask |= m

    masks = [GenericMask(m, m.shape[0], m.shape[1]) for m in masks]

    polygons = [poly.reshape(-1, 2) for mask in masks for poly in mask.polygons]
    return polygons, mask


def perform_watershed(mask, min_distance=5):
    distances = ndimage.distance_transform_edt(mask)
    local_max = peak_local_max(distances, indices=False, min_distance=min_distance,
                               labels=mask)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    # markers = ndimage.label(local_max)[0]
    labels = watershed(-distances, markers, mask=mask)
    polys = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros_like(mask, dtype=np.uint8)
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c.squeeze() for c in cnts if c.shape[0] > 2]
        if cnts:
            c = max(cnts, key=lambda cnt: Polygon(cnt.squeeze()).area)
            polys += [c.squeeze()]

    return polys


def create_crops_list(border_size, crop_size, image):
    crop_coords = create_crops_coords_list(crop_size, border_size, image)
    crops = [image[i:i + crop_size, j:j + crop_size, ...] for (i, j) in crop_coords]
    return list(zip(crops, crop_coords))


def create_crops_coords_list(crop_size, border_size, image):
    vert = list(range(0, image.shape[0], crop_size - 2 * border_size))
    horiz = list(range(0, image.shape[1], crop_size - 2 * border_size))
    vert = list(filter(lambda v: v + crop_size <= image.shape[0], vert)) + [image.shape[0] - crop_size]
    horiz = list(filter(lambda v: v + crop_size <= image.shape[1], horiz)) + [image.shape[1] - crop_size]
    crop_coords = list(itertools.product(vert, horiz))
    return crop_coords


class ExperimentImagesPredictor(DirWatcher):
    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, parent_structs, connectivity_dir,
                 _processor_number, cell_model, crop_size, border_size, device, threshold):
        super().__init__(input_dir, process_dir, output_dir, f'experiment-images-predictor-{_processor_number}')
        self.threshold = threshold
        self.device = device
        if device == 'cuda:':
            self.device += str(_processor_number)
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
        try:
            with open(f'{directory}/ratios.pickle', 'rb') as f:
                ratios = pickle.load(f)
        except FileNotFoundError:
            ratios = {s: (64, 64) for s in bboxes.keys()}
        sections = sorted([s for s in bboxes.keys() if bboxes[s]])
        for section in sorted(sections):
            self.process_section(directory, experiment_id, section, bboxes[section], mask[:, :, section], ratios[section])

    def process_section(self, directory, experiment_id, section, bboxes, mask, ratios):
        self.logger.info(f"Experiment {experiment_id}: processing section {section}...")
        for bbox in bboxes:
            x, y, w, h = bbox
            x = x * ratios[0]
            w = w * ratios[0]
            y = y * ratios[1]
            h = h * ratios[1]
            cellmask_fname = f'{directory}/cellmask-{experiment_id}-{section}-{x}_{y}_{w}_{h}.npz'
            if not os.path.isfile(cellmask_fname):
                image = cv2.imread(f'{directory}/full-{experiment_id}-{section}-{x}_{y}_{w}_{h}.jpg',
                                   cv2.IMREAD_GRAYSCALE)
                polys = self.predict_cells(image, mask, x, y, ratios)
                np.savez(cellmask_fname, polys)
            else:
                self.logger.info(f"{cellmask_fname} already exists, skipping segmentation...")

    def predict_cells(self, image, section_mask, x, y, ratios):
        crops = create_crops_list(self.border_size, self.crop_size, image)
        cell_mask = np.zeros_like(image)
        mask_basex, mask_crop_sizex = map(lambda v: v // ratios[0], [x, self.crop_size])
        mask_basey, mask_crop_sizey = map(lambda v: v // ratios[1], [y, self.crop_size])
        for crop, coords in crops:
            ym, xm = coords[0] // ratios[1] + mask_basey, coords[1] // ratios[0] + mask_basex
            if section_mask[ym: min(ym + mask_crop_sizey + 1, section_mask.shape[0] - 1),
               xm: min(xm + mask_crop_sizex + 1, section_mask.shape[1] - 1)].sum() > 0:
                outputs = self.cell_model(cv2.cvtColor(image[coords[0]: coords[0] + self.crop_size,
                                          coords[1]: coords[1] + self.crop_size], cv2.COLOR_GRAY2BGR))
                _, mask = extract_predictions(outputs["instances"].to("cpu"))
                if mask is not None:
                    cell_mask[coords[0]: coords[0] + self.crop_size, coords[1]: coords[1] + self.crop_size] = \
                        np.logical_or(cell_mask[coords[0]: coords[0] + self.crop_size,
                                      coords[1]: coords[1] + self.crop_size], mask)

        polys = perform_watershed(cell_mask)
        return mask


class ExperimentPredictorTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment downloader")

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument('--cell_model', action='store', required=True, help='Cell segmentation model')
        parser.add_argument('--crop_size', default=160, type=int, action='store', help='Size of a single crop')
        parser.add_argument('--border_size', default=10, type=int, action='store',
                            help='Size of the border (to make crops overlap)')
        parser.add_argument('--device', default='cuda', action='store', help='Model execution device')
        parser.add_argument('--threshold', default=0.5, action='store', help='Prediction threshold')

    def prepare_input(self, connectivity_dir, **kwargs):
        mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        mcc.get_structure_tree()

    def execute_task(self, structs, structure_map_dir, **kwargs):
        predictor = ExperimentImagesPredictor(parent_structs=ast.literal_eval(structs),
                                               structure_map_dir=structure_map_dir, **kwargs)
        experiments = os.listdir(structure_map_dir)
        predictor.run_until_count(len(experiments))


if __name__ == '__main__':
    dl = ExperimentPredictorTaskManager()
    dl.run()

# python experiment_images_predictor.py -i output/full_brain/downloaded -d output/full_brain/pd_proc -o output/full_brain/predicted -c mouse_connectivity -m output/sectiondata/data/result -s "[997]" -p2 --cell_model output/new_cells/model_0324999.pth
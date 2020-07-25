import json
import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import GenericMask

from process_full_scan import split_image

source = """
{
    "_via_attributes": {
        "file": {},
        "region": {}
    },
    "_via_settings": {
        "core": {
            "buffer_size": "18",
            "default_filepath": "",
            "filepath": {
                ".": 1
            }
        },
        "project": {
            "name": "via_project_19Jan2020_11h0m"
        },
        "ui": {
            "annotation_editor_fontsize": 0.8,
            "annotation_editor_height": 25,
            "image": {
                "on_image_annotation_editor_placement": "NEAR_REGION",
                "region_color": "__via_default_region_color__",
                "region_label": "__via_region_id__",
                "region_label_font": "10px Sans"
            },
            "image_grid": {
                "img_height": 80,
                "rshape_fill": "none",
                "rshape_fill_opacity": 0.3,
                "rshape_stroke": "yellow",
                "rshape_stroke_width": 2,
                "show_image_policy": "all",
                "show_region_shape": true
            },
            "leftsidebar_width": 18
        }
    }
}
"""

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.MODEL.DEVICE = 'cpu'

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/Users/david/Desktop/model_0449999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.DATASETS.TEST = ("balloon_val",)

def create_file_entry(file_path, annotations):
    regions = []
    for annotation in annotations:
        all_points_x = []
        all_points_y = []
        for (x, y) in annotation.tolist():
            all_points_x += [x]
            all_points_y += [y]

        regions += [{
            "region_attributes": {},
            "shape_attributes": {
                "all_points_x": all_points_x,
                "all_points_y": all_points_y,
                "name": "polygon"
            }
        }]

    current = {
        "file_attributes": {},
        "filename": os.path.basename(file_path),
        "regions": regions,
        "size": os.path.getsize(file_path)
    }

    return current


def predict(crop, predictor):
    outputs = predictor(crop)
    predictions = outputs["instances"].to("cpu")
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    if not predictions.has('pred_masks'):
        return None

    masks = np.asarray(predictions.pred_masks)
    masks = [GenericMask(x, 312, 312) for x in masks]
    polygons = [poly.reshape(-1, 2) for mask in masks for poly in mask.polygons]
    return polygons


def concat_eq(crop):
    return np.concatenate(
        [cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR), crop], axis=1)


def build_project(crops, outImages, process, suffix):
    os.makedirs(outImages, exist_ok=True)
    project = json.loads(source)
    metadata = dict()
    for crop, coords, polygons in crops:
        processed = process(crop)
        file_name = f'{coords[0]}_{coords[1]}_{suffix}.jpg'
        image_path = os.path.join(outImages, file_name)
        cv2.imwrite(image_path, processed)
        entry = create_file_entry(image_path, polygons)
        entry['size'] = os.path.getsize(image_path)
        metadata[entry['filename'] + str(entry['size'])] = entry
    project['_via_img_metadata'] = metadata
    with open(os.path.join(outImages, 'via.json'), 'wt') as out:
        json.dump(project, out, indent=4, sort_keys=True)


def file_without_ext(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def main():
    full_scans = [f for f in listdir('data/full-scans') if isfile(join('data/full-scans', f))]
    predictor = DefaultPredictor(cfg)
    for full_scan in full_scans:
        full_scan = os.path.join('data/full-scans', full_scan)
        crops, image = split_image(full_scan, 312, 20)
        crops = [(crop, coords, predict(crop, predictor)) for crop, coords in crops[:3]]

        suffix = file_without_ext(full_scan)
        build_project(crops, f'data/test/{suffix}/simple', lambda c: c, suffix)
        build_project(crops, f'data/test/{suffix}/augmented', concat_eq, suffix)


if __name__ == '__main__':
    main()

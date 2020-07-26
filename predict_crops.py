import argparse
import os
import pickle
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import GenericMask

from process_full_scan import split_image


def initialize_model(model_name, device, threshold):
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
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("balloon_val",)
    return DefaultPredictor(cfg)


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


def main(model, full_dir, output, crop_size, border_size, device='cuda', threshold=0.5):
    predictor = initialize_model(model, device, threshold)
    full_scans = [f for f in listdir(full_dir) if isfile(join(full_dir, f))]
    splitted = [(fs, split_image(os.path.join(full_dir, fs), crop_size, border_size)[0]) for fs in full_scans]
    predictions = {fs: [(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), coords, predict(crop, predictor)) for crop, coords in
                        split[:3]] for fs, split in splitted}
    with open(output, 'wb') as f:
        pickle.dump(predictions, f)
    print(f'Predictions saved to {output}. Exiting...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detectron Mask R-CNN for cells segmentation - predictions')
    parser.add_argument('--model', required=True, action='store', help='Model name')
    parser.add_argument('--full_dir', required=True, action='store', help='Directory that contains full scans')
    parser.add_argument('--output', required=True, action='store', help='Filename for predictions pickle')
    parser.add_argument('--crop_size', default=312, type=int, action='store', help='Size of a single crop')
    parser.add_argument('--border_size', default=20, type=int, action='store',
                        help='Size of the border (to make crops overlap)')
    parser.add_argument('--device', default='cuda', action='store', help='Model execution device')
    parser.add_argument('--threshold', default=0.5, action='store', help='Prediction threshold')
    args = parser.parse_args()

    main(args.model, args.full_dir, args.output, args.crop_size, args.border_size, args.device, args.threshold)

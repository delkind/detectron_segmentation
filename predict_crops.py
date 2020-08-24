import argparse
import itertools
import os
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import GenericMask

from predict_experiment import initialize_model, create_crops_list


def split_image(image_path, crop_size, border_size):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return create_crops_list(border_size, crop_size, image), image


def predict(fs, crop, coords, predictor):
    outputs = predictor(crop)
    print(f'Processed {fs}:{str(coords)}')
    return outputs["instances"].to("cpu")


def extract_masks(predictions):
    return np.asarray(predictions.pred_masks)


def process_prediction(full_scan, predictions, crop_size):
    mask = np.zeros_like(cv2.imread(full_scan, cv2.IMREAD_GRAYSCALE))
    for coords, pred in predictions:
        crop_masks = extract_masks(pred)
        crop_mask = np.zeros((crop_size, crop_size), dtype=bool)
        for i in range(crop_masks.shape[0]):
            crop_mask |= crop_masks[i, :, :]
        mask[coords[0]: coords[0] + crop_size, coords[1]: coords[1] + crop_size] |= crop_mask
    return mask


def main(model, full_dir, output, crop_size, border_size, device='cuda', threshold=0.5):
    predictor = initialize_model(model, device, threshold)
    full_scans = [f for f in listdir(full_dir) if isfile(join(full_dir, f))]
    splitted = [(fs, split_image(os.path.join(full_dir, fs), crop_size, border_size)[0]) for fs in full_scans]
    predictions = {fs: process_prediction(os.path.join(full_dir, fs),
                                          [(coords, predict(fs, crop, coords, predictor)) for crop, coords in split],
                                          crop_size)
                   for fs, split in splitted}

    with open(output, 'wb') as f:
        pickle.dump(predictions, f)
    print(f'Predictions saved to {output}. Exiting...')


def process_predict_arguments():
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
    return parser.parse_args()


if __name__ == '__main__':
    args = process_predict_arguments()
    main(args.model, args.full_dir, args.output, args.crop_size, args.border_size, args.device, args.threshold)

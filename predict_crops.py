import argparse
import itertools
import os
import pickle
from os import listdir
from os.path import isfile, join

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def split_image(image_path, crop_size, border_size):
    image = cv2.imread(image_path)
    return create_crops_list(border_size, crop_size, image), image


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


def predict(fs, crop, coords, predictor):
    outputs = predictor(crop)
    print(f'Processed {fs}:{str(coords)}')
    return outputs["instances"].to("cpu")


def main(model, full_dir, output, crop_size, border_size, device='cuda', threshold=0.5):
    predictor = initialize_model(model, device, threshold)
    full_scans = [f for f in listdir(full_dir) if isfile(join(full_dir, f))]
    splitted = [(fs, split_image(os.path.join(full_dir, fs), crop_size, border_size)[0]) for fs in full_scans]
    predictions = {fs: [(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), coords, predict(fs, crop, coords, predictor))
                        for crop, coords in split[:3]] for fs, split in splitted}
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

import argparse
import os
import pickle
from os import listdir
from os.path import isfile, join

import cv2

from predict_crops import initialize_model, split_image
from process_predicted_crops import to_polygons, file_without_ext


def predict(fs, crop, coords, predictor):
    outputs = predictor(crop)
    print(f'Processed {fs}:{str(coords)}')
    return to_polygons(outputs["instances"].to("cpu"))


def main(model, full_dir, output, crop_size, border_size, device='cuda', threshold=0.5):
    predictor = initialize_model(model, device, threshold)
    full_scans = [f for f in listdir(full_dir) if isfile(join(full_dir, f))]
    splitted = [(fs, split_image(os.path.join(full_dir, fs), crop_size, border_size)[0]) for fs in full_scans]
    predictions = {fs: [(coords, predict(fs, crop, coords, predictor)) for crop, coords in split] for fs, split in splitted}
    os.makedirs(output, exist_ok=True)
    for fs in full_scans:
        image = cv2.imread(os.path.join(full_dir, fs), cv2.IMREAD_COLOR)
        for coords, poly in predictions[fs]:
            cv2.polylines(image[coords[0]: coords[0] + crop_size, coords[1]:coords[1] + crop_size, :],
                          poly, isClosed=True, thickness=1, color=(0, 255, 0))
        fname = file_without_ext(fs)
        cv2.imwrite(os.path.join(output, fname + '.jpg'), image)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detectron Mask R-CNN for cells segmentation - predictions (full image)')
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

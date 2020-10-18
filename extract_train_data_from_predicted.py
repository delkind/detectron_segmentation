import argparse
import json
import os
import pickle
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from detectron2.utils.visualizer import GenericMask

from predict_experiment import create_crops_coords_list
from rect import Rect

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


def get_eq_func(images):
    bins_edges_min_max = [0, 256]
    num_bins = 256
    bin_count = np.zeros(256, dtype=float)
    for image in images:
        bin_count_img, _ = np.histogram(image, num_bins, bins_edges_min_max)
        bin_count += bin_count_img

    pdf = bin_count / np.sum(bin_count)
    cdf = np.cumsum(pdf)
    f_eq = np.round(cdf * 255).astype(int)
    return f_eq


def create_file_entry(file_path, polygons):
    regions = []
    for annotation in polygons:
        all_points_x = []
        all_points_y = []
        for (x, y) in annotation.tolist():
            all_points_x += [x]
            all_points_y += [y]

        if len(set(all_points_x)) == 1 or len(set(all_points_y)) == 1:
            continue

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


def build_project(crops, output_dir, process, suffix, false_positive):
    os.makedirs(output_dir, exist_ok=True)
    project = json.loads(source)
    metadata = dict()
    for crop, coords, prediction in crops:
        processed = process(crop)
        file_name = f'{coords[1]}_{coords[0]}_{suffix}.jpg'
        image_path = os.path.join(output_dir, file_name)
        cv2.imwrite(image_path, processed)
        entry = create_file_entry(image_path, to_polygons(prediction) if not false_positive else [])
        entry['size'] = os.path.getsize(image_path)
        metadata[entry['filename'] + str(entry['size'])] = entry
    project['_via_img_metadata'] = metadata
    with open(os.path.join(output_dir, 'via.json'), 'wt') as out:
        json.dump(project, out, indent=4, sort_keys=True)


def file_without_ext(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def concat_eq(crop, eq_func):
    return np.concatenate([eq_func[crop], crop], axis=1)


def to_polygons(predictions):
    if type(predictions) == np.ndarray:
        masks = [GenericMask(predictions, predictions.shape[0], predictions.shape[1])]
    else:
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        if not predictions.has('pred_masks'):
            return None

        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, x.shape[0], x.shape[1]) for x in masks]

    polygons = [poly.reshape(-1, 2) for mask in masks for poly in mask.polygons]
    return polygons


def create_annotated_scan(original, mask, output_dir, suffix):
    original = original.copy()
    mask = GenericMask(mask, *mask.shape)
    polygons = [poly.reshape(-1, 2) for poly in mask.polygons]
    cv2.polylines(original, polygons, isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.imwrite(os.path.join(output_dir, f'{suffix}.jpg'), original)
    return cv2.imread(os.path.join(output_dir, f'{suffix}.jpg'))


def coords_to_str(crop):
    return f'{crop[0][1][1]}_{crop[0][1][0]}-{crop[-1][1][1]}_{crop[-1][1][0]}'
    pass


def group_by_rows(crops):
    ycoords = set(crop[1][0] for crop in crops)
    sorted_coords = sorted([crop[1] for crop in crops if crop[1][0] == val] for val in ycoords)
    crops = {crop[1]: crop for crop in crops}
    crops = [[crops[coord] for coord in coord_list] for coord_list in sorted_coords]
    return crops


def create_rois_list(window_title, crops, full_image, project_size=10):
    shrink_factor = 5
    resized = cv2.resize(full_image, (0, 0), fx=1/shrink_factor, fy=1/shrink_factor)
    rois = cv2.selectROIs(window_title, resized)
    cv2.destroyAllWindows()
    if type(rois) is not tuple:
        rois = [Rect(*roi) for roi in (rois * shrink_factor).tolist()]
        crops = [[crop for crop in crops if Rect(crop[1][1], crop[1][0], crop[0].shape[1], crop[0].shape[0]).
            intersection(r).area() > 0] for r in rois]
        return crops
    else:
        return []


def create_non_overlapping_crops(image, mask, crop_size):
    crop_coords = create_crops_coords_list(crop_size, 0, image)
    crops = [image[i:i + crop_size, j:j + crop_size, ...] for (i, j) in crop_coords]
    masks = [mask[i:i + crop_size, j:j + crop_size, ...] for (i, j) in crop_coords]
    return list(zip(crops, crop_coords, masks))


def main(output_dir, input_dir, slice, select_roi, false_positive, crop_size):
    results = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    results = {'-'.join(f.split('-')[:2]) for f in results if slice is None or f.split('-')[1] == slice}
    results = sorted(list(results))

    os.makedirs(output_dir, exist_ok=True)
    for suffix in results:
        original_file_name = os.path.join(input_dir, suffix+'-original.jpg')
        mask_file_name = os.path.join(input_dir, suffix+'-cellmask.png')
        if not isfile(original_file_name):
            continue
        print(f"Processing {original_file_name}")
        original = cv2.cvtColor(cv2.imread(original_file_name, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        mask = cv2.imread(mask_file_name, cv2.IMREAD_GRAYSCALE)
        ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, ctrs, color=255)
        mask = mask.astype(bool)
        eq_func = get_eq_func([original])

        crops = create_non_overlapping_crops(original, mask, crop_size)
        full_image = create_annotated_scan(original, mask, output_dir, suffix)

        if select_roi:
            crops = create_rois_list(suffix, crops, full_image)
            for num, crop in enumerate(crops):
                build_project(crop, f'{output_dir}/{suffix}/simple/{num}', lambda c: c, suffix, false_positive)
                build_project(crop, f'{output_dir}/{suffix}/augmented/{num}', lambda c: concat_eq(c, eq_func),
                              suffix, false_positive)
        else:
            crops = group_by_rows(crops)
            for crop in crops:
                build_project(crop, f'{output_dir}/{suffix}/simple/{crop[0][1][0]}', lambda c: c, suffix,
                              False)
                build_project(crop, f'{output_dir}/{suffix}/augmented/{crop[0][1][0]}',
                              lambda c: concat_eq(c, eq_func), suffix, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detectron Mask R-CNN for cells segmentation - create train data from predictions')
    parser.add_argument('--input_dir', required=True, action='store', help='Directory containing full scan files')
    parser.add_argument('--slice', default=None, action='store', help='Directory containing full scan files')
    parser.add_argument('--output_dir', required=True, action='store', help='Directory to output the predicted results')
    parser.add_argument('--crop_size', default=312, type=int, action='store', help='Size of a single crop')
    parser.add_argument('--select_roi', action='store_true', help='Manually select ROIs')
    parser.add_argument('--false_positive', action='store_true', help='Manually select ROIs')
    args = parser.parse_args()

    main(**vars(args))

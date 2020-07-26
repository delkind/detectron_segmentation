import argparse
import json
import os
import pickle

import cv2
import numpy as np
from detectron2.utils.visualizer import GenericMask

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


def build_project(crops, output_dir, process, suffix):
    os.makedirs(output_dir, exist_ok=True)
    project = json.loads(source)
    metadata = dict()
    for crop, coords, prediction in crops:
        processed = process(crop)
        file_name = f'{coords[1]}_{coords[0]}_{suffix}.jpg'
        image_path = os.path.join(output_dir, file_name)
        cv2.imwrite(image_path, processed)
        entry = create_file_entry(image_path, to_polygons(prediction))
        entry['size'] = os.path.getsize(image_path)
        metadata[entry['filename'] + str(entry['size'])] = entry
    project['_via_img_metadata'] = metadata
    with open(os.path.join(output_dir, 'via.json'), 'wt') as out:
        json.dump(project, out, indent=4, sort_keys=True)


def file_without_ext(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def concat_eq(crop):
    return np.concatenate([cv2.equalizeHist(crop), crop], axis=1)


def to_polygons(predictions):
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


def create_annotated_scan(crops, output_dir, suffix):
    coord_lists = list(zip(*[coords for _, coords, _ in crops]))
    size_y = max(coord_lists[0]) + 312
    size_x = max(coord_lists[1]) + 312
    original = np.zeros((size_y, size_x, 3), dtype=np.int16)
    polygons = []
    for crop, coords, pred in crops:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        polygons += [poly + list(reversed(coords)) for poly in to_polygons(pred)]
        original[coords[0]:coords[0] + 312, coords[1]: coords[1] + 312, :] = crop

    cv2.polylines(original, polygons, isClosed=True, color=(0, 255, 0))
    cv2.imwrite(os.path.join(output_dir, f'{suffix}.jpg'), original)
    return original


def coords_to_str(crop):
    return f'{crop[0][1][1]}_{crop[0][1][0]}-{crop[-1][1][1]}_{crop[-1][1][0]}'
    pass


def group_by_rows(crops):
    ycoords = set(crop[1][0] for crop in crops)
    sorted_coords = sorted([crop[1] for crop in crops if crop[1][0] == val] for val in ycoords)
    crops = {crop[1]: crop for crop in crops}
    crops = [[crops[coord] for coord in coord_list] for coord_list in sorted_coords]
    return crops


def main(predictions, output_dir, select_roi):
    with open(predictions, 'rb') as f:
        predictions = pickle.load(f)
        os.makedirs(output_dir, exist_ok=True)
        for full_scan, crops in predictions.items():
            suffix = file_without_ext(full_scan)
            full_image = create_annotated_scan(crops, output_dir, suffix)
            if not select_roi:
                crops = group_by_rows(crops)
                for crop in crops:
                    build_project(crop, f'{output_dir}/{suffix}/simple/{crop[0][1][0]}', lambda c: c, suffix)
                    build_project(crop, f'{output_dir}/{suffix}/augmented/{crop[0][1][0]}', concat_eq, suffix)
            else:
                resized = cv2.resize(full_image, (0, 0), fx=0.0125, fy=0.0125)
                # r = cv2.selectROI("Select region to edit", resized)
                # r = cv2.selectROI("Select region to edit", crops[0][0])
                # crops = [crop for crop in crops if ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detectron Mask R-CNN for cells segmentation - parse predictions')
    parser.add_argument('--output_dir', required=True, action='store', help='Directory to output the predicted results')
    parser.add_argument('--predictions', required=True, action='store', help='Predictions pickle')
    parser.add_argument('--select_roi', action='store_true', help='Manually select ROIs')
    args = parser.parse_args()

    main(args.predictions, args.output_dir, args.select_roi)

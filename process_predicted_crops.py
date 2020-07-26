import argparse
import json
import os
import pickle

import cv2
import numpy as np


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


def concat_eq(crop):
    return np.concatenate(
        [cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR), crop], axis=1)


def main(predictions, output_dir):
    with open(predictions, 'rb') as f:
        predictions = pickle.load(f)
        for full_scan, crops in predictions.items():
            suffix = file_without_ext(full_scan)
            build_project(crops, f'{output_dir}/{suffix}/simple', lambda c: c, suffix)
            build_project(crops, f'{output_dir}/{suffix}/augmented', concat_eq, suffix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detectron Mask R-CNN for cells segmentation - parse predictions')
    parser.add_argument('--output_dir', required=True, action='store', help='Directory to output the predicted results')
    parser.add_argument('--predictions', required=True, action='store', help='Predictions pickle')
    args = parser.parse_args()

    main(args.predictions, args.output_dir)

import argparse
import ast
import json

import cv2
import numpy as np

from process_predicted_crops import create_file_entry
from rect import Rect


def edit_hippo_dataset(data_dir, images_to_edit, input_project, output_project, include_selected):
    filenames_to_edit = {f'{exp_id}-{slice_id}' for exp_id, rng in
                         ast.literal_eval(images_to_edit).items() for slice_id in rng} \
        if images_to_edit is not None else None
    project = json.load(open(input_project, 'rt'))
    entries = project['_via_img_metadata']
    new_entries = dict()
    for num, (name, entry) in enumerate(entries.items()):
        print(f"Processing image {num + 1} of {len(entries)}")
        if filenames_to_edit is None or entry["filename"].split('.')[0] in filenames_to_edit:
            image_path = f'{data_dir}/{entry["filename"]}'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            contours = [
                np.asarray(list(zip(r['shape_attributes']['all_points_x'], r['shape_attributes']['all_points_y'])))
                for r in entry['regions']]
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, contours, color=1)
            while True:
                image_to_display = image.copy()
                image_to_display[mask != 0] = np.minimum(image_to_display.astype(np.uint16)[mask != 0] * 4, 255).astype(np.uint8)
                image_to_display = cv2.cvtColor(image_to_display, cv2.COLOR_GRAY2BGR)
                image_to_display[:, :, 0][mask == 0] = 0
                image_to_display[:, :, 1][mask == 0] = 0
                image_to_display = cv2.resize(image_to_display, (0, 0), fx=3, fy=3)
                roi = cv2.selectROI("Select region to edit", image_to_display)
                print(roi)
                roi = Rect(*(np.asarray(list(roi)) // 3))
                if roi.area() == 0:
                    break
                mask[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w] = 1 if include_selected else 0

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c.squeeze() for c in contours if c.shape[0] > 1 and cv2.boundingRect(c)[3] > 50]
            new_entries[f'{entry["filename"]}{entry["size"]}'] = create_file_entry(f'{data_dir}/{entry["filename"]}',
                                                                                   contours)
        else:
            new_entries[name] = entry
    project['_via_img_metadata'] = new_entries
    json.dump(project, open(output_project, 'wt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detectron Mask R-CNN for cells segmentation - experiment prediction')
    parser.add_argument('--data_dir', required=True, action='store', help='Dataset directory')
    parser.add_argument('--input_project', required=True, action='store', help='Input project file name')
    parser.add_argument('--output_project', required=True, action='store', help='Output project file name')
    parser.add_argument('--images_to_edit', default=None, action='store', help='List of images to edit')
    parser.add_argument('--include_selected', default=False, action='store_true',
                        help='Include or exclude selection from mask')

    args = parser.parse_args()
    print(vars(args))
    edit_hippo_dataset(**vars(args))

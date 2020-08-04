import json
import os
import numpy as np
import matplotlib.pyplot as plt
from rect import Rect

import cv2

from process_predicted_crops import create_file_entry

data_dir = 'data/train_hippo'

project = json.load(open(f'{data_dir}/annotations1.json', 'rt'))
entries = project['_via_img_metadata']
new_entries = dict()

images_to_edit = {613898292: [70], 627869431: [70]}

filenames_to_edit = {f'{exp_id}-{slice_id}' for exp_id, rng in images_to_edit.items() for slice_id in rng}

for name, entry in entries.items():
    if entry["filename"].split('.')[0] in filenames_to_edit:
        image_path = f'{data_dir}/{entry["filename"]}'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        contours = [np.asarray(list(zip(r['shape_attributes']['all_points_x'], r['shape_attributes']['all_points_y'])))
                    for r in entry['regions']]
        pattern = np.zeros_like(image)
        cv2.fillPoly(pattern, contours, color=1)
        while True:
            roi = cv2.selectROI("Select region to edit", cv2.resize(pattern * image, (0, 0), fx=3, fy=3))
            print(roi)
            roi = Rect(*(np.asarray(list(roi)) // 3))
            if roi.area() == 0:
                break
            pattern[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w] = 1

        contours, _ = cv2.findContours(pattern, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c.squeeze() for c in contours if c.shape[0] > 1 and cv2.boundingRect(c)[3] > 50]
        new_entries[f'{entry["filename"]}{entry["size"]}'] = create_file_entry(f'{data_dir}/{entry["filename"]}',
                                                                               contours)
    else:
        new_entries[name] = entry

project['_via_img_metadata'] = new_entries
json.dump(project, open(f'{data_dir}/annotations2.json', 'wt'))

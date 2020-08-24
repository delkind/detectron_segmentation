import json
import os

import cv2
import numpy as np

from process_predicted_crops import create_file_entry
from rect import Rect

input_dir = 'scan_thumbnails'
output_dir = 'data/train_hippo'

project = json.load(open(f'{input_dir}/annotations.json', 'rt'))
entries = {e[0]: e[1] for e in project['_via_img_metadata'].items() if e[1]['regions']}
del project['_via_data_format_version']
del project['_via_image_id_list']
os.makedirs(output_dir, exist_ok=True)
new_entries = dict()
for entry in entries.values():
    image_path = f'{input_dir}/{entry["filename"]}'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(f'{output_dir}/{entry["filename"]}', image)

    shape = entry['regions'][0]['shape_attributes']
    mask = np.zeros_like(image)
    ones = np.ones((shape['height'], shape['width']))
    mask[shape['y']: shape['y'] + shape['height'], shape['x']: shape['x'] + shape['width']] = ones
    image = cv2.equalizeHist(image)
    image = image * mask
    ret, thresh = cv2.threshold(image, 80, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c.squeeze() for c in contours if c.shape[0] > 1 and cv2.boundingRect(c)[3] > 50]
    pattern = np.zeros_like(image)
    cv2.fillPoly(pattern, contours, color=1)
    while True:
        roi = cv2.selectROI("Select region to edit", cv2.resize(pattern * image, (0, 0), fx=3, fy=3))
        print(roi)
        roi = Rect(*(np.asarray(list(roi))//3))
        if roi.area() == 0:
            break
        pattern[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w] = 0

    contours, _ = cv2.findContours(pattern, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c.squeeze() for c in contours if c.shape[0] > 1 and cv2.boundingRect(c)[3] > 50]
    new_entries[f'{entry["filename"]}{entry["size"]}'] = create_file_entry(f'{output_dir}/{entry["filename"]}',
                                                                           contours)

project['_via_img_metadata'] = new_entries
json.dump(project, open(f'{output_dir}/annotations.json', 'wt'))
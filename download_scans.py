import json
import math
import os
import urllib.request
from collections import defaultdict

import cv2
from allensdk.api.queries.image_download_api import ImageDownloadApi
from skimage import io

input_dir = 'scan_thumbnails/'
project_name = 'scan_thumbnails/annotations.json'
output_dir = 'data/full-scans'
zoom = 2

project = json.load(open(project_name, 'rt'))

entries = [e for e in project['_via_img_metadata'].values() if e['regions']]
scan_info = [
    {
        'experiment_id': int(entry['filename'].split('.')[0].split('-')[0]),
        'slice_id': int(entry['filename'].split('.')[0].split('-')[1]),
        'shape': entry['regions'][0]['shape_attributes'],
        'filename': entry['filename']
    } for entry in entries
]

experiments = defaultdict(list)

for info in scan_info:
    experiments[info['experiment_id']] += [info]

image_api = ImageDownloadApi()

for experiment_id, infos in experiments.items():
    images = {s['section_number']: s for s in image_api.section_image_query(experiment_id)}
    for info in infos:
        img = images[info["slice_id"]]
        y = int(info["shape"]["y"]) // 4
        x = int(info["shape"]["x"]) // 4
        width = int(info["shape"]["width"]) // 4
        height = int(info["shape"]["height"]) // 4
        zoom_factor = 8
        print(f'\tProcessing slice {info["slice_id"]}')
        url = f'http://connectivity.brain-map.org/cgi-bin/imageservice?path={img["path"]}&zoom={zoom_factor}&'\
              f'top={y*2**8}&left={x*2**8}&width={width*2**zoom_factor}&'\
              f'height={height*2**zoom_factor}&filter=range&filterVals=0,1051,0,0,0,0'
        urllib.request.urlretrieve(url, f'{output_dir}/{experiment_id}-{img["section_number"]}.jpg')
        image = cv2.imread(f'{output_dir}/{experiment_id}-{img["section_number"]}.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f'{output_dir}/{experiment_id}-{img["section_number"]}.jpg', image)

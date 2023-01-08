import itertools
import os
import pickle
import random
import sys

import skimage.transform
import skimage.io
import skimage.exposure
import numpy as np
import pandas as pd
import re
import cv2
from tqdm import tqdm

from experiment_images_downloader import ExperimentImagesDownloader
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


TILE_SIZE = 320


def create_crops_coords_list(crop_size, border_size, image):
    vert = list(range(0, image.shape[0], crop_size[0] - 2 * border_size))
    horiz = list(range(0, image.shape[1], crop_size[1] - 2 * border_size))
    vert = list(filter(lambda v: v + crop_size[0] <= image.shape[0], vert)) + [image.shape[0] - crop_size[0]]
    horiz = list(filter(lambda v: v + crop_size[1] <= image.shape[1], horiz)) + [image.shape[1] - crop_size[1]]
    crop_coords = list(itertools.product(vert, horiz))
    return crop_coords


def get_all_children(tree, reg_id):
    children = eval(tree.loc[reg_id].region_ids)
    all_children = []
    for c in children:
        if c != reg_id:
            all_children += [c] + get_all_children(tree, c)

    return list(set(all_children))


def extract_bounding_boxes(dilation_kernel, mask, area_threshold=0):
    bboxes = ExperimentImagesDownloader.get_bounding_boxes(mask)
    bbmask = np.zeros_like(mask, dtype=np.uint8)
    for bbox in bboxes:
        cv2.rectangle(bbmask, *bbox.corners(), color=1, thickness=-1)
    bbmask = cv2.dilate(bbmask, dilation_kernel)
    bboxes = [bbox for bbox in ExperimentImagesDownloader.get_bounding_boxes(bbmask) if bbox.area() > area_threshold]
    return bboxes


def main(path, output_dir, section_data_dir):
    print(f"Processing {path}...")
    tree = pd.read_csv(f"{path}/141-2-001_Atlas_Regions.csv").set_index(['id'])

    brains = [d for d in os.listdir(path) if os.path.isdir(f"{path}/{d}/")]
    mcc = MouseConnectivityCache(manifest_file=f'/tmp/mouse_connectivity_manifest.json')
    acronyms = mcc.get_structure_tree().get_id_acronym_map()
    acronyms = {k.lower(): v for k, v in acronyms.items()}
    allen_ids = {iid: acronyms[acro.lower()] for iid, acro in tree.acronym.to_dict().items()}
    id_mapper = np.vectorize(lambda p: allen_ids[p] if p != 0 else 0)

    for b in brains:
        print(f"Processing {b}...")
        spl = re.split('_|-', b)
        dir_name = ''.join(spl[4:8] + [spl[10]])
        os.makedirs(f'{output_dir}/{dir_name}', exist_ok=True)
        os.makedirs(f'{section_data_dir}/{dir_name}', exist_ok=True)
        sections = [s for s in os.listdir(f"{path}/{b}/") if s.endswith("ch03.tif")]
        section_numbers = [int((re.findall(r'_S\d+_', s) + re.findall(r'_S\d+\.', s))[0][2:-1]) for s in sections]
        sections = dict(zip(section_numbers, sections))
        atlas_dirs = [s for s in os.listdir(f"{path}/{b}/") if
                      s.startswith("atlas") and os.path.isdir(f"{path}/{b}/{s}")]
        atlas_items = [s for s in os.listdir(f"{path}/{b}/{atlas_dirs[0]}")
                       if os.path.isfile(f"{path}/{b}/{atlas_dirs[0]}/{s}")]
        atlas_numbers = [int((re.findall(r'_S\d+_', s) + re.findall(r'_S\d+\.', s))[0][2:-1]) for s in atlas_items]
        atlas_items = dict(zip(atlas_numbers, atlas_items))
        available = sorted(list(set(section_numbers).intersection(set(atlas_numbers))))
        atlas_shape = skimage.io.imread(f'{path}/{b}/{atlas_dirs[0]}/{atlas_items[available[0]]}').shape
        atlas_array = [np.zeros(atlas_shape, dtype=np.int)]
        for i in tqdm(range(available[-1]), "Processing atlas..."):
            if i in atlas_items:
                raw_image = skimage.io.imread(f'{path}/{b}/{atlas_dirs[0]}/{atlas_items[i]}')
                if raw_image.sum() == 0:
                    print(f"Invalid atlas file: {dir_name}-{i}")
                    atlas = np.zeros(atlas_shape, dtype=int)
                else:
                    atlas = id_mapper(raw_image.astype(int))
            else:
                atlas = np.zeros(atlas_shape, dtype=int)

            atlas_array += [atlas]

        atlas_array = np.stack(atlas_array, axis=2)
        np.savez(f'{section_data_dir}/{dir_name}/{dir_name}-sections.npz', atlas_array)

        bbox_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        bboxes = {section: extract_bounding_boxes(bbox_dilation_kernel, atlas_array[:, :, section]) for section in available}
        bboxes = {k: v for k, v in bboxes.items() if v}

        with open(f'{output_dir}/{dir_name}/bboxes.pickle', 'wb') as f:
            pickle.dump(bboxes, f)

        ratios = dict()

        for i, bbox in tqdm(bboxes.items(), "Processing images..."):
            image = skimage.io.imread(f'{path}/{b}/{sections[i]}')
            image = (image / 255).astype(np.uint8)

            ratio_y, ratio_x = (np.array(image.shape) // np.array(atlas_array[:, :, i].shape)).tolist()
            ratios[i] = (ratio_y, ratio_x)

            thumbnail = cv2.resize(image, (image.shape[0] // 64, image.shape[1] // 64))
            cv2.imwrite(f'{output_dir}/{dir_name}/thumbnail-{dir_name}-{i}.jpg', thumbnail)

            for bb in bbox:
                x, y, w, h = bb
                crop = image[y * ratio_y: (y + h) * ratio_y, x * ratio_x: (x + w) * ratio_x]
                cv2.imwrite(f'{output_dir}/{dir_name}/full-{dir_name}-{i}-{x * ratio_x}_{y * ratio_y}_{w * ratio_y}_{h * ratio_y}.jpg', crop)

        with open(f'{output_dir}/{dir_name}/ratios.pickle', 'wb') as f:
            pickle.dump(ratios, f)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

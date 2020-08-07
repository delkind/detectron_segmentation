import argparse
import itertools
import os
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
from allensdk.api.queries.image_download_api import ImageDownloadApi
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import GenericMask
from tqdm import tqdm


def download_url(url, decription=None, filename=None):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if filename is not None and os.path.isfile(filename):
        return filename

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=decription) as t:
        filename, _ = urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
        return filename


def extract_predictions(predictions):
    if not predictions.has('pred_masks'):
        return None, None

    masks = np.asarray(predictions.pred_masks)
    if masks.shape[0] == 0:
        return None, None

    mask = np.zeros_like(masks[0, :, :])
    for m in masks:
        mask |= m

    masks = [GenericMask(m, m.shape[0], m.shape[1]) for m in masks]

    polygons = [poly.reshape(-1, 2) for mask in masks for poly in mask.polygons]
    return polygons, mask


def predict_hippo(image, predictor):
    outputs = predictor(cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
    polygons, mask = extract_predictions(outputs["instances"].to("cpu"))
    if polygons is None or mask is None:
        return None, None, None
    boxes = [cv2.boundingRect(p) for p in polygons]
    boxes = [[x, y, x + w, y + h] for x, y, w, h in boxes]
    zipped = list(zip(*boxes))
    xl, yl, xr, yr = zipped
    bbox = [min(xl), min(yl), max(xr), max(yr)]
    return polygons, bbox, mask


def create_file_name(cache_dir, image_desc, suffix):
    dataset_id = image_desc["data_set_id"]
    section_number = image_desc["section_number"]
    if cache_dir:
        filename = f'{cache_dir}/{section_number}-{suffix}.jpg'
    else:
        filename = None
    return dataset_id, filename, section_number


def download_thumbnail(image_desc, cache_dir):
    dataset_id, filename, section_number = create_file_name(cache_dir, image_desc, 'thumbnail')
    return download_section_image(image_desc["path"], 2,
                                  f'Downloading thumbnail for experiment {dataset_id} '
                                  f'section {section_number}', filename=filename)


def download_full_scan(image_desc, box, cache_dir):
    dataset_id, filename, section_number = create_file_name(cache_dir, image_desc, 'full')
    dataset_id = image_desc["data_set_id"]
    return download_section_image(image_desc["path"], 8,
                                  f'Downloading full scan for experiment {dataset_id} '
                                  f'section {section_number}', box,
                                  filename=filename)


def download_section_image(image_path, zoom, description, box=None, filename=None):
    url = f'http://connectivity.brain-map.org/cgi-bin/imageservice?path={image_path}&' \
          f'mime=1&zoom={zoom}&&filter=range&filterVals=0,534,0,1006,0,4095'
    if box is not None:
        x1, y1, x2, y2 = box
        url += f'&top={y1}&left={x1}&width={x2 - x1}&height={y2 - y1}'
    temp_file = download_url(url, decription=description, filename=filename)
    image = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    if filename is None:
        os.remove(temp_file)
    return image


def calculate_areas(image, mask, bbox):
    mask_area = mask.sum()
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    total_area = image.shape[1] * image.shape[0]
    cbox = (bbox_area * 10000 // total_area) / 10000
    cmask = (mask_area * 10000 // bbox_area) / 10000
    btm = (cmask * 10000 // cbox) / 10000
    return cbox, cmask, btm


def annotate_thumbnail(experiment, section, image, bbox, polygons, cbox, cmask, btm):
    def put_text(text, bottom):
        textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (image.shape[1] - textsize[0]) // 2
        if bottom:
            text_y = (image.shape[0] - textsize[1] - 5)
        else:
            text_y = textsize[1] + 5
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness,
                    bottomLeftOrigin=False)

    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    put_text(f'Experiment {experiment}, section {section}', False)
    put_text(f'bbox coverage: {cbox}, mask coverage: {cmask}, btm: {btm}', True)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=1)
    cv2.polylines(image, polygons, isClosed=True, thickness=1, color=(0, 255, 0))
    return image


def process_thumbnail(annotated_thumbnail_callback, btm_range, experiment_id, hippo_predictor, images, section,
                      cache_dir):
    proceed = False
    thumbnail = download_thumbnail(images[section], cache_dir)
    polygons, bbox, mask = predict_hippo(thumbnail, hippo_predictor)
    if polygons is not None:
        cbox, cmask, btm = calculate_areas(thumbnail, mask, bbox)
        if btm_range[0] <= btm <= btm_range[1]:
            xl, yl, xr, yr = bbox
            bbox = [xl - 5, yl - 5, xr + 5, yr + 5]
            thumbnail = annotate_thumbnail(experiment_id, section, thumbnail, bbox, polygons, cbox, cmask, btm)
            if annotated_thumbnail_callback is not None:
                annotated_thumbnail_callback(thumbnail, proceed)
            proceed = True
        else:
            print(f"Skipping section {section}: BTM is not in range")
    else:
        print(f"Skipping section {section}: no hippocampus found")
    return proceed, bbox, mask


def plot_thumbnail(thumbnail):
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
    plt.show()


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


def create_annotated_scan(original, cell_mask, hippo_mask, filename_prefix, save_mask):
    annotated = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    gmask = GenericMask(cell_mask, *cell_mask.shape)
    polygons = [poly.reshape(-1, 2) for poly in gmask.polygons]
    gmask = GenericMask(hippo_mask, *hippo_mask.shape)
    polygons += [poly.reshape(-1, 2) for poly in gmask.polygons]
    cv2.polylines(annotated, polygons, isClosed=True, color=(0, 255, 0), thickness=3)
    annotated = cv2.resize(annotated, (0, 0), fx=0.25, fy=0.25)
    cv2.imwrite(f'{filename_prefix}-annotated.jpg', annotated)
    if save_mask:
        if cell_mask is not None:
            cv2.imwrite(f'{filename_prefix}-cellmask.png', cell_mask.astype(np.uint8) * 255)
        cv2.imwrite(f'{filename_prefix}-hippomask.png', hippo_mask.astype(np.uint8) * 255)
        cv2.imwrite(f'{filename_prefix}-original.jpg', original)


def predict_cells(border_size, cell_predictor, crop_size, experiment_id, hippo_mask, image, section):
    crops = create_crops_list(border_size, crop_size, image)
    cell_mask = np.zeros_like(hippo_mask)
    for crop, coords in tqdm(crops, desc=f"Predicting crops for {experiment_id}-{section}"):
        outputs = cell_predictor(cv2.cvtColor(image[coords[0]: coords[0] + crop_size,
                                              coords[1]: coords[1] + crop_size],
                                              cv2.COLOR_GRAY2BGR))
        _, mask = extract_predictions(outputs["instances"].to("cpu"))
        cell_mask[coords[0]: coords[0] + crop_size, coords[1]: coords[1] + crop_size] = \
            np.logical_or(cell_mask[coords[0]: coords[0] + crop_size, coords[1]: coords[1] + crop_size], mask)
    mask = np.logical_and(hippo_mask, cell_mask)
    return mask


def obtain_full_scan(bbox, cache_dir, images, mask, section):
    x1, y1, x2, y2 = bbox
    hippo_mask = cv2.resize(mask[y1:y2, x1:x2].astype(np.uint8), (0, 0), fx=64, fy=64).astype(bool)
    bbox = np.asarray(bbox) * 64
    image = cv2.cvtColor(download_full_scan(images[section], bbox, cache_dir), cv2.COLOR_BGR2GRAY)
    return hippo_mask, image


def predict_experiment(annotated_thumbnail_callback, border_size, cache, cell_predictor, crop_size, experiment_id,
                       hippo_predictor, image_api, max_btm, max_section, min_btm, min_section, output_dir, save_mask):
    images = image_api.section_image_query(experiment_id)
    images = {i['section_number']: i for i in images}
    if cache:
        cache_dir = f'{output_dir}/cache/{experiment_id}'
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = None
    output_dir = f'{output_dir}/{experiment_id}'
    os.makedirs(output_dir, exist_ok=True)
    for section_num, section in enumerate(range(min_section, max_section + 1)):
        proceed, bbox, mask = process_thumbnail(annotated_thumbnail_callback, (min_btm, max_btm),
                                                experiment_id, hippo_predictor,
                                                images, section, cache_dir)
        if proceed:
            hippo_mask, image = obtain_full_scan(bbox, cache_dir, images, mask, section)

            if cell_predictor is not None:
                cell_mask = predict_cells(border_size, cell_predictor, crop_size, experiment_id, hippo_mask, image,
                                          section)
            else:
                cell_mask = None

            create_annotated_scan(image, cell_mask, hippo_mask, f'{output_dir}/{experiment_id}-{section}',
                                  save_mask)


def main(experiment_ids, min_section, max_section, hippo_predictor,
         cell_predictor, min_btm, max_btm, crop_size, border_size,
         output_dir, cache=False, bbox_padding=0, device='cuda',
         threshold=0.5, save_mask=False,
         annotated_thumbnail_callback=None):
    hippo_predictor = initialize_model(hippo_predictor, device, 0.5)
    if cell_predictor is not None:
        cell_predictor = initialize_model(cell_predictor, device, threshold)
    image_api = ImageDownloadApi()
    for experiment_id in experiment_ids.split(','):
        predict_experiment(annotated_thumbnail_callback, border_size, cache, cell_predictor, crop_size,
                           int(experiment_id), hippo_predictor, image_api, max_btm, max_section, min_btm, min_section,
                           output_dir, save_mask)


def initialize_model(model_path, device, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return DefaultPredictor(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detectron Mask R-CNN for cells segmentation - experiment prediction')
    parser.add_argument('--experiment_ids', '-e', required=True, action='store', help='Experiment ID')
    parser.add_argument('--cell_predictor', '-c', default=None, action='store', help='Cell model path')
    parser.add_argument('--hippo_predictor', '-p', required=True, action='store', help='Hippocampus model path')
    parser.add_argument('--output_dir', '-o', required=True, action='store', help='Directory that will contain output')

    parser.add_argument('--min_section', default=63, action='store', help='Minimum section number to analyze')
    parser.add_argument('--max_section', default=83, action='store', help='Maximum section number to analyze')
    parser.add_argument('--min_btm', default=5.5, action='store', help='Minimum box-to-mask ratio to analyze')
    parser.add_argument('--max_btm', default=13.6, action='store', help='Maximum box-to-mask ratio to analyze')
    parser.add_argument('--bbox_padding', default=0, action='store', help='Padding (in pixels) for the hippocampus bounding box')
    parser.add_argument('--cache', default=False, action='store_true', help='Cache the downloaded crops')
    parser.add_argument('--save_mask', default=True, action='store_true', help='Cache the downloaded crops')

    parser.add_argument('--crop_size', default=312, type=int, action='store', help='Size of a single crop')
    parser.add_argument('--border_size', default=20, type=int, action='store',
                        help='Size of the border (to make crops overlap)')
    parser.add_argument('--device', default='cuda', action='store', help='Model execution device')
    parser.add_argument('--threshold', default=0.5, action='store', help='Prediction threshold')
    args = parser.parse_args()

    print(vars(args))
    main(**vars(args))

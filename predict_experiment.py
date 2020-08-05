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


def download_thumbnail(image_desc):
    return download_section_image(image_desc["path"], 2,
                                  f'Downloading thumbnail for experiment {image_desc["data_set_id"]} '
                                  f'section {image_desc["section_number"]}')


def download_full_scan(image_desc, box):
    dataset_id = image_desc["data_set_id"]
    section_number = image_desc["section_number"]
    return download_section_image(image_desc["path"], 8,
                                  f'Downloading full scan for experiment {dataset_id} '
                                  f'section {section_number}', box,
                                  filename=f'{dataset_id}-{section_number}.jpg')


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


def annotate_image(experiment, section, image, bbox, polygons, cbox, cmask, btm):
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


def process_thumbnail(annotated_thumbnail_callback, btm_range, experiment_id, hippo_predictor, images, section):
    proceed = False
    thumbnail = download_thumbnail(images[section])
    polygons, bbox, mask = predict_hippo(thumbnail, hippo_predictor)
    if polygons is not None:
        cbox, cmask, btm = calculate_areas(thumbnail, mask, bbox)
        if btm_range[0] <= btm <= btm_range[1]:
            xl, yl, xr, yr = bbox
            bbox = [xl - 5, yl - 5, xr + 5, yr + 5]
            thumbnail = annotate_image(experiment_id, section, thumbnail, bbox, polygons, cbox, cmask, btm)
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


def create_annotated_scan(original, mask, filename):
    original = original.copy()
    mask = GenericMask(mask, *mask.shape)
    polygons = [poly.reshape(-1, 2) for poly in mask.polygons]
    cv2.polylines(original, polygons, isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.imwrite(filename, original)


def predict_experiment(experiment_id, relevant_sections, hippo_predictor, cell_predictor, btm_range,
                       crop_size, border_size, bbox_padding=0, annotated_thumbnail_callback=None):
    image_api = ImageDownloadApi()
    images = image_api.section_image_query(experiment_id)
    images = {i['section_number']: i for i in images}

    for section_num, section in enumerate(relevant_sections):
        proceed, bbox, mask = process_thumbnail(annotated_thumbnail_callback, btm_range, experiment_id, hippo_predictor,
                                                images, section)
        if proceed:
            x1, y1, x2, y2 = bbox
            hippo_mask = cv2.resize(mask[y1:y2, x1:x2].astype(np.uint8), (0, 0), fx=64, fy=64).astype(bool)
            cell_mask = np.zeros_like(hippo_mask)
            bbox = np.asarray(bbox) * 64
            image = cv2.cvtColor(download_full_scan(images[section], bbox), cv2.COLOR_BGR2GRAY)
            crops = create_crops_list(border_size, crop_size, image)
            for num, (crop, coords) in enumerate(crops):
                print(f'Predicting crop {num} out of {len(crops)}...')
                outputs = cell_predictor(cv2.cvtColor(image[coords[0]: coords[0] + crop_size,
                                                      coords[1]: coords[1] + crop_size],
                                                      cv2.COLOR_GRAY2BGR))
                _, mask = extract_predictions(outputs["instances"].to("cpu"))
                cell_mask[coords[0]: coords[0] + crop_size, coords[1]: coords[1] + crop_size] = \
                    np.logical_or(cell_mask[coords[0]: coords[0] + crop_size, coords[1]: coords[1] + crop_size], mask)

            mask = np.logical_and(hippo_mask, cell_mask)
            create_annotated_scan(image, mask, f'{experiment_id}-{section}-annotated.jpg')


def initialize_model(model_path, device, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return DefaultPredictor(cfg)


if __name__ == '__main__':
    hippo_predictor = initialize_model('output/model_final.pth', 'cuda', 0.5)
    cell_predictor = initialize_model('output/model_cells.pth', 'cuda', 0.5)
    predict_experiment(129564675, range(67, 83), hippo_predictor, cell_predictor, (5.5, 13.6), 312, 20)

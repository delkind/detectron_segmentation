import argparse
import ast
import http.client
import os
import pickle
import shutil
import time
import urllib.error
import urllib.request

import PIL.Image
import cv2
import numpy as np
import simplejson
from PIL import Image
from allensdk.api.queries.image_download_api import ImageDownloadApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager
from rect import Rect

PIL.Image.MAX_IMAGE_PIXELS = None


class ExperimentImagesDownloader(DirWatcher):
    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, structs, connectivity_dir,
                 _processor_number, brightness_threshold, strains):
        super().__init__(input_dir, process_dir, output_dir, f'experiment-images-downloader-{_processor_number}')
        self.brightness_threshold = brightness_threshold
        self.structs = ast.literal_eval(structs)
        self.segmentation_dir = structure_map_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        struct_tree = self.mcc.get_structure_tree()
        structure_ids = [i for sublist in struct_tree.descendant_ids(self.structs) for i in sublist]
        self.structure_ids = set(structure_ids)
        self.image_api = ImageDownloadApi()
        self.bbox_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
        exps = self.mcc.get_experiments(dataframe=True)
        items = []
        for s in strains:
            males = exps[(exps.strain == s) & (exps.gender == 'M')].id.tolist()
            females = exps[(exps.strain == s) & (exps.gender == 'F')].id.tolist()
            min_len = min(len(males), len(females))
            males = sorted(males[:min_len])
            females = sorted(females[:min_len])
            items += [str(i) for j in zip(males, females) for i in j]
        self.initial_items = [i for i in items if i in self.initial_items] + [i for i in self.initial_items
                                                                              if i not in items]

    def on_process_error(self, item, exception):
        retval = super().on_process_error(item, exception)
        self.logger.error(f"Error occurred during processing", exc_info=True)
        if any(map(lambda x: issubclass(type(exception), x), [urllib.error.HTTPError, OSError, ValueError, http.client.error])):
            return False
        else:
            return retval

    def process_item(self, item, directory):
        experiment_id = int(item)
        retries = 0
        images = []
        while True:
            try:
                time.sleep(2 ** (retries // 2))
                images = self.image_api.section_image_query(experiment_id)
                break
            except simplejson.errors.JSONDecodeError as e:
                if retries > 10:
                    raise e
                else:
                    self.logger.info(f"Exception invoking image API, retrying")
                    retries += 1
                    continue

        images = {i['section_number']: i for i in images}
        segmentation = np.load(f'{self.segmentation_dir}/{item}/{item}-sections.npz')['arr_0']
        mask = np.isin(segmentation, list(self.structure_ids))
        locs = np.where(mask)
        sections = [s for s in sorted(np.unique(locs[2]).tolist()) if s in images]
        bboxes = {section: self.extract_bounding_boxes(mask[:, :, section]) for section in sections}
        valid_sections = list(filter(lambda s: bboxes[s], sections))
        brightness = self.calculate_brightness(bboxes, directory, experiment_id, images, valid_sections)

        if brightness < self.brightness_threshold:
            return False

        with open(f'{directory}/bboxes.pickle', 'wb') as f:
            pickle.dump(bboxes, f)

        for section in valid_sections:
            self.process_section(directory, experiment_id, images, section, bboxes[section])

    def calculate_brightness(self, bboxes, directory, experiment_id, images, valid_sections):
        brightness = 0
        for section in valid_sections:
            self.download_snapshot(experiment_id, section, images[section], directory)
            filename = f'{directory}/thumbnail-{experiment_id}-{section}.jpg'
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            total = 0
            count = 0
            for x, y, w, h in bboxes[section]:
                crop = image[y: y + h, x: x + w]
                pixels = crop[crop != 0]
                total += pixels.sum()
                count += pixels.shape[0]

            brightness += total / count
        brightness /= len(valid_sections)
        return brightness

    def process_section(self, directory, experiment_id, images, section, bboxes):
        self.logger.info(f"Experiment {experiment_id}, downloading section {section}...")
        self.download_snapshot(experiment_id, section, images[section], directory)
        for bbox in bboxes:
            self.download_fullres(experiment_id, section, bbox, images[section], directory)

    def extract_bounding_boxes(self, mask, area_threshold=0):
        bboxes = self.get_bounding_boxes(mask)
        bbmask = np.zeros_like(mask, dtype=np.uint8)
        for bbox in bboxes:
            cv2.rectangle(bbmask, *bbox.corners(), color=1, thickness=-1)
        bbmask = cv2.dilate(bbmask, self.bbox_dilation_kernel)
        bboxes = [bbox for bbox in self.get_bounding_boxes(bbmask) if bbox.area() > area_threshold]
        return bboxes

    @staticmethod
    def get_bounding_boxes(mask):
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)[-2:]
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 5:
                rects.append(Rect(x=x, y=y, w=w, h=h))
        return rects

    def download_fullres(self, experiment_id, section, bbox, image_desc, directory):
        url = f'https://connectivity.brain-map.org/cgi-bin/imageservice?path={image_desc["path"]}&' \
              f'mime=1&zoom={8}&&filter=range&filterVals=0,534,0,1006,0,4095'
        x, y, w, h = bbox.scale(64)
        url += f'&top={y}&left={x}&width={w}&height={h}'
        filename = f'{directory}/full-{experiment_id}-{section}-{x}_{y}_{w}_{h}.jpg'
        for retries in range(3):
            fname, _, downloaded = self.retrieve_url(filename, url)
            if downloaded:
                try:
                    image = Image.open(fname)
                    image.load()
                    break
                except OSError or FileNotFoundError as e:
                    os.remove(fname)
                    if retries == 2:
                        raise e
                    else:
                        self.logger.info(f"Corrupted file {fname}, re-downloading {filename}")
            else:
                self.logger.info(f"Cached version of {filename} used, skipping verification")

        return filename

    def download_snapshot(self, experiment_id, section, image_desc, directory):
        url = f'https://connectivity.brain-map.org/cgi-bin/imageservice?path={image_desc["path"]}&' \
              f'mime=1&zoom={2}&&filter=range&filterVals=0,534,0,1006,0,4095'
        filename = f'{directory}/thumbnail-{experiment_id}-{section}.jpg'
        filename, _, _ = self.retrieve_url(filename, url)
        return filename

    def download_brightness_snapshot(self, experiment_id, section, image_desc, directory):
        url = f'https://connectivity.brain-map.org/cgi-bin/imageservice?path={image_desc["path"]}&' \
              f'mime=1&zoom={2}&&filter=range&filterVals=0,534,0,1006,0,4095'
        filename = f'{directory}/thumbnail-{experiment_id}-{section}.jpg'
        filename, _, _ = self.retrieve_url(filename, url)
        return filename

    def retrieve_url(self, filename, url, retries=10):
        if os.path.isfile(filename):
            self.logger.info(f"File {filename} already downloaded")
            return filename, None, False

        backoff = 0
        urllib.request.urlcleanup()
        while True:
            try:
                time.sleep(2 ** backoff)
                fname, msg = urllib.request.urlretrieve(url, filename=f'{filename}.partial')
                os.replace(fname, filename)
                return filename, msg, True
            except http.client.HTTPException or OSError or urllib.error.HTTPError as e:
                backoff += 1
                retries -= 1
                if retries > 0:
                    self.logger.info(f"Transient error downloading {url}, "
                                     f"retrying ({retries} retries left) ...", exc_info=True)
                    continue
                else:
                    self.logger.exception(f"Retry count exceeded or permanent error for {url} ({filename}), exiting...")
                    raise e


class ExperimentDownloadTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment downloader")

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument('--brightness_threshold', '-b', action='store', default=30, type=int,
                            help='Experiment brightness threshold')
        parser.add_argument('--strains', action='store', default=('C57BL/6J', 'FVB.CD1(ICR)'),
                            help='Experiment brightness threshold')

    def prepare_input(self, input_dir, connectivity_dir, structure_map_dir, **kwargs):
        mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        mcc.get_structure_tree()
        experiments = os.listdir(structure_map_dir)
        try:
            os.makedirs(input_dir)
            for e in experiments:
                os.makedirs(f'{input_dir}/{e}')
        except FileExistsError:
            pass
            # for e in experiments:
            #     if os.path.isdir(f'{input_dir}/{e}'):
            #         shutil.rmtree(f'{input_dir}/{e}')
            #         os.makedirs(f'{input_dir}/{e}')

    def execute_task(self, **kwargs):
        downloader = ExperimentImagesDownloader(**kwargs)
        downloader.run_until_empty()


if __name__ == '__main__':
    dl = ExperimentDownloadTaskManager()
    dl.run()

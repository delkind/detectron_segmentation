import ast
import os
import pickle
import shutil
import urllib.error
import urllib.request

import cv2
import numpy as np
from allensdk.api.queries.image_download_api import ImageDownloadApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from dir_watcher import DirWatcher
from experiment_process_task_manager import ExperimentProcessTaskManager
from rect import Rect


class ExperimentImagesDownloader(DirWatcher):
    def __init__(self, input_dir, process_dir, output_dir, structure_map_dir, structs, connectivity_dir,
                 _processor_number):
        super().__init__(input_dir, process_dir, output_dir, f'experiment-images-downloader-{_processor_number}')
        self.structs = ast.literal_eval(structs)
        self.segmentation_dir = structure_map_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        struct_tree = self.mcc.get_structure_tree()
        structure_ids = [i for sublist in struct_tree.descendant_ids(self.structs) for i in sublist]
        self.structure_ids = set(structure_ids)
        self.image_api = ImageDownloadApi()
        self.bbox_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))

    def process_item(self, item, directory):
        experiment_id = int(item)
        images = self.image_api.section_image_query(experiment_id)
        images = {i['section_number']: i for i in images}
        segmentation = np.load(f'{self.segmentation_dir}/{item}/{item}-sections.npz')['arr_0']
        mask = np.isin(segmentation, list(self.structure_ids))
        locs = np.where(mask)
        sections = [s for s in sorted(np.unique(locs[2]).tolist()) if s in images]
        bboxes = {section: self.extract_bounding_boxes(mask[:, :, section]) for section in sections}
        with open(f'{directory}/bboxes.pickle', 'wb') as f:
            pickle.dump(bboxes, f)
        for section in filter(lambda s: bboxes[s], sections):
            self.process_section(directory, experiment_id, images, section, bboxes[section])

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
        url = f'http://connectivity.brain-map.org/cgi-bin/imageservice?path={image_desc["path"]}&' \
              f'mime=1&zoom={8}&&filter=range&filterVals=0,534,0,1006,0,4095'
        x, y, w, h = bbox.scale(64)
        url += f'&top={y}&left={x}&width={w}&height={h}'
        filename = f'{directory}/full-{experiment_id}-{section}-{x}_{y}_{w}_{h}.jpg'
        filename, _ = self.retrieve_url(filename, url)
        return filename

    def download_snapshot(self, experiment_id, section, image_desc, directory):
        url = f'http://connectivity.brain-map.org/cgi-bin/imageservice?path={image_desc["path"]}&' \
              f'mime=1&zoom={2}&&filter=range&filterVals=0,534,0,1006,0,4095'
        filename = f'{directory}/thumbnail-{experiment_id}-{section}.jpg'
        filename, _ = self.retrieve_url(filename, url)
        return filename

    def retrieve_url(self, filename, url, retries=100):
        if os.path.isfile(filename):
            self.logger.debug(f"File {filename} already downloaded")
            return filename, None

        while True:
            try:
                fname, msg = urllib.request.urlretrieve(url, filename=f'{filename}.partial')
                os.replace(fname, filename)
                return filename, msg
            except urllib.error.HTTPError as e:
                if 500 <= e.code < 600:
                    retries = retries - 1
                    if retries > 0:
                        self.logger.debug(f"Transient error downloading {url}, "
                                          f"retrying ({retries} retries left) ...", exc_info=e)
                        continue
                    else:
                        self.logger.exception(f"Retry count exceeded for {url} ({filename}), exiting...")
                        raise e


class ExperimentDownloadTaskManager(ExperimentProcessTaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment downloader")

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

import argparse
import ast
import os
import shutil
import urllib.request

import cv2
import numpy as np
from allensdk.api.queries.image_download_api import ImageDownloadApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from dir_watcher import DirWatcher
from rect import Rect
from task_manager import TaskManager


class ExperimentImagesDownloader(DirWatcher):
    def __init__(self, input_dir, intermediate_dir, results_dir, segmentation_dir, parent_structs, mcc_dir, number):
        super().__init__(input_dir, intermediate_dir, results_dir, f'experiment-images-downloader-{number}')
        self.parent_structs = parent_structs
        self.segmentation_dir = segmentation_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'{mcc_dir}/mouse_connectivity_manifest.json')
        struct_tree = self.mcc.get_structure_tree()
        structure_ids = [i for sublist in struct_tree.descendant_ids(self.parent_structs) for i in sublist]
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
        sections = sorted(np.unique(locs[2]).tolist())
        for section in sections:
            bboxes = self.extract_bounding_boxes(mask[:, :, section])
            if not bboxes:
                if section > 75:
                    break
                else:
                    continue
            self.logger.info(f"Experiment {experiment_id}, downloading section {section}...")
            self.download_snapshot(experiment_id, section, bboxes, images[section], directory)
            for bbox in bboxes:
                if bbox.w > 5 and bbox.h > 5:
                    self.download_section(experiment_id, section, bbox, images[section], directory)

    def extract_bounding_boxes(self, mask):
        bboxes = self.get_bounding_boxes(mask)
        bbmask = np.zeros_like(mask, dtype=np.uint8)
        for bbox in bboxes:
            cv2.rectangle(bbmask, *bbox.corners(), color=1, thickness=-1)
        bbmask = cv2.dilate(bbmask, self.bbox_dilation_kernel)
        bboxes = self.get_bounding_boxes(bbmask)
        return bboxes

    @staticmethod
    def get_bounding_boxes(mask):
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)[-2:]
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 5:
                rects.append(Rect(x=x, y=y, w=w, h=h))
        return rects

    @staticmethod
    def download_section(experiment_id, section, bbox, image_desc, directory):
        url = f'http://connectivity.brain-map.org/cgi-bin/imageservice?path={image_desc["path"]}&' \
              f'mime=1&zoom={8}&&filter=range&filterVals=0,534,0,1006,0,4095'
        x, y, w, h = bbox
        x, y, w, h = list(map(lambda a: a * 64, [x, y, w, h]))
        url += f'&top={y}&left={x}&width={w}&height={h}'
        filename = f'{directory}/full-{experiment_id}-{section}-{x}_{y}_{w}_{h}.jpg'
        filename, _ = urllib.request.urlretrieve(url, filename=filename)

    @staticmethod
    def download_snapshot(experiment_id, section, bboxes, image_desc, directory):
        url = f'http://connectivity.brain-map.org/cgi-bin/imageservice?path={image_desc["path"]}&' \
              f'mime=1&zoom={2}&&filter=range&filterVals=0,534,0,1006,0,4095'
        filename = f'{directory}/thumbnail-{experiment_id}-{section}.jpg'
        filename, _ = urllib.request.urlretrieve(url, filename=filename)
        image = cv2.imread(filename)
        for bbox in bboxes:
            if bbox.w > 5 and bbox.h > 5:
                cv2.rectangle(image, *bbox.corners(), color=(0, 255, 0), thickness=2)
        cv2.imwrite(filename, image)


class ExperimentDownloadTaskManager(TaskManager):
    def __init__(self):
        super().__init__("Connectivity experiment downloader")

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument('--input-dir', '-i', action='store', required=True, help='Input directory')
        parser.add_argument('--process_dir', '-d', action='store', required=True, help='Processing directory')
        parser.add_argument('--output_dir', '-o', action='store', required=True,
                            help='Results output directory')
        parser.add_argument('--connectivity_dir', '-c', action='store', required=True,
                            help='Connectivity cache directory')
        parser.add_argument('--structure_map_dir', '-m', action='store', required=True,
                            help='Connectivity cache directory')
        parser.add_argument('--structs', '-s', action='store', required=True,
                            help='List of structures to process')

    def prepare_input(self, connectivity_dir, input_dir, process_dir, output_dir, structure_map_dir, structs):
        mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json')
        struct_tree = mcc.get_structure_tree()
        experiments = os.listdir(structure_map_dir)
        try:
            os.makedirs(input_dir)
            for e in experiments:
                os.makedirs(f'{input_dir}/{e}')
        except FileExistsError:
            for e in experiments:
                if os.path.isdir(f'{input_dir}/{e}'):
                    shutil.rmtree(f'{input_dir}/{e}')
                    os.makedirs(f'{input_dir}/{e}')

    def execute_task(self, connectivity_dir, input_dir, process_dir, output_dir, structs, structure_map_dir):
        downloader = ExperimentImagesDownloader(input_dir, process_dir, output_dir, structure_map_dir,
                                                ast.literal_eval(structs), connectivity_dir, self.process_number)
        downloader.run_until_empty()


if __name__ == '__main__':
    dl = ExperimentDownloadTaskManager()
    dl.run()

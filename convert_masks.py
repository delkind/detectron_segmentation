import argparse
from os import listdir
from os.path import isfile, join
import numpy as np

import cv2
from tqdm import tqdm

from predict_experiment import get_downloaded_experiments


def verify_conversion(ctrs, mask):
    print(f"Source contours shape: {len(ctrs)}")
    trial_mask = np.zeros_like(mask)
    ctrs1, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(trial_mask, ctrs, color=(0, 255, 0, 255))
    ctrs1, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Resulting contours shape: {len(ctrs1)}")
    return not (len(ctrs) != len(ctrs1) or any(map(lambda c: (c[0] != c[1]).any(), zip(ctrs, ctrs1))))


def main(input_dir, verify):
    experiment_ids = get_downloaded_experiments(input_dir)
    for experiment_id in experiment_ids:
        input_dir = f'output/experiments/{experiment_id}'
        results = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
        results = sorted(list({'-'.join(f.split('-')[:2]) for f in results}))
        for result in tqdm(results, desc=f"Processing {experiment_id}..."):
            filename = f'{input_dir}/{result}-mask.png'
            if not isfile(filename):
                continue
            mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(np.stack([np.zeros_like(mask), mask, np.zeros_like(mask), mask], axis=2),
                                 dtype=np.uint8)
            cv2.polylines(mask, ctrs, True, color=(0, 255, 0, 255))
            if not verify_conversion(ctrs, mask):
                print(f"Mismatch in {filename}")
            cv2.imwrite(filename, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detectron Mask R-CNN for cells segmentation - convert mask file')
    parser.add_argument('input_dir', action='store', help='Directory containing full scan files')
    parser.add_argument('--verify', action='store_true', help='Verify shapes match')
    args = parser.parse_args()
    main(**vars(args))

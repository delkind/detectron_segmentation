import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

SMOOTH = 0.0001


def main():
    data_dir = './data/test_set'
    predictions_file = 'via.json'
    gt_file = 'ground_truth.json'

    perform_evaluation(data_dir, gt_file, predictions_file)


def calculate_score(gt_mask, pred_mask, score):
    return (score(gt_mask != 0, pred_mask != 0) + score(gt_mask == 0, pred_mask == 0)) / 2


def intersection(gt, pred):
    return (gt & pred).sum()


def iou(gt, pred):
    union_cell = (gt | pred).sum()
    return (intersection(gt, pred) + SMOOTH) / (union_cell + SMOOTH)


def f1(gt, pred):
    return (2 * intersection(gt, pred) + SMOOTH) / ((gt.sum() + pred.sum()) + SMOOTH)


def total_error(gt, pred):
    return (gt.sum() + pred.sum() - 2 * intersection(gt, pred)) / (2 * gt.shape[0] * gt.shape[1])


score_calcs = {'IoU (Jaccard Index)': iou, 'F1 (Dice)': f1, 'Total errors': total_error}


def perform_evaluation(data_dir, gt_file, predictions_file):
    pred_proj = json.load(open(f'{data_dir}/{predictions_file}', 'rt'))
    gt_proj = json.load(open(f'{data_dir}/{gt_file}', 'rt'))
    assert sorted(gt_proj['_via_img_metadata'].keys()) == sorted(pred_proj['_via_img_metadata'].keys())
    files = sorted(gt_proj['_via_img_metadata'].keys())
    scores = defaultdict(dict)
    for f in files:
        gt_mask = to_mask(data_dir, gt_proj['_via_img_metadata'][f])
        pred_mask = to_mask(data_dir, pred_proj['_via_img_metadata'][f])
        for k, v in score_calcs.items():
            scores[f][k] = calculate_score(gt_mask, pred_mask, v)
    regions = list(set(s.split('-')[0] for s in scores.keys()))
    reg_scores = pd.DataFrame({r: {n: np.mean([scores[f][n] for f in scores.keys() if f.startswith(r)])
                                   for n in score_calcs.keys()} for r in regions}).transpose()
    pass


def to_mask(data_dir, desc):
    cnts = []
    mask = np.zeros_like(cv2.imread(f'{data_dir}/{desc["filename"]}', cv2.IMREAD_GRAYSCALE))
    for r in desc['regions']:
        coords = [[x, y] for x, y in zip(r['shape_attributes']['all_points_x'],
                                         r['shape_attributes']['all_points_y'])]
        coords = coords + [coords[0]]
        cnts += [np.array(coords)]
    cv2.fillPoly(mask, cnts, color=1)
    return mask


if __name__ == '__main__':
    main()

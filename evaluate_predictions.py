import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SMOOTH = 0.0001


def main():
    data_dir = './data/test_set'
    predictions_file = 'via.json'
    gt_file = 'ground_truth.json'
    pred_proj = json.load(open(f'{data_dir}/{predictions_file}', 'rt'))
    gt_proj = json.load(open(f'{data_dir}/{gt_file}', 'rt'))
    return perform_evaluation(data_dir, gt_proj, pred_proj)


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


def scores_for_region(r, scores):
    relevant_keys = [f for f in scores.keys() if f.startswith(r)]
    result = {n: np.mean([scores[f][n] for f in relevant_keys]) for n in score_calcs.keys()}
    result = {**result,
              **{"Cell accuracy": np.mean([scores[f]['TP']/(scores[f]['TP'] + scores[f]['FN']) for f in relevant_keys])},
              **{"Cell FP Rate": np.mean([scores[f]['FP']/(scores[f]['TP'] + scores[f]['FN']) for f in relevant_keys])},
              **{"Total cells": np.sum([(scores[f]['TP'] + scores[f]['FN']) for f in relevant_keys])}
              }

    return result


def perform_evaluation(data_dir, gt_proj, pred_proj):
    assert sorted(gt_proj['_via_img_metadata'].keys()) == sorted(pred_proj['_via_img_metadata'].keys())
    files = sorted(gt_proj['_via_img_metadata'].keys())
    scores = defaultdict(dict)
    for f in tqdm(files):
        gt_mask = to_mask(data_dir, gt_proj['_via_img_metadata'][f])
        pred_mask = to_mask(data_dir, pred_proj['_via_img_metadata'][f])
        for k, v in score_calcs.items():
            scores[f][k] = calculate_score(gt_mask, pred_mask, v)

        cell_result = analyze_cells(gt_mask, pred_mask)
        scores[f]['TP'] = cell_result[3]
        scores[f]['FN'] = cell_result[1]
        scores[f]['FP'] = cell_result[2]
    regions = list(set(s.split('-')[0] for s in scores.keys()))
    reg_scores = pd.DataFrame({r: scores_for_region(r, scores) for r in regions}).transpose()
    return scores


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


def analyze_cells(gt_mask, pred_mask):
    composite_mask = gt_mask + pred_mask * 2
    cnts, _ = cv2.findContours((composite_mask != 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = defaultdict(int)
    for c in cnts:
        result[classify_cell(c, composite_mask)] += 1

    return result


def classify_cell(c, composite_mask):
    candidate_mask = np.zeros_like(composite_mask)
    cv2.fillPoly(candidate_mask, [c], color=1, lineType=8)
    cell_layout = composite_mask[candidate_mask.astype(bool)]

    if (c.shape[0] / cell_layout.shape[0]) > 0.8:
        return 0

    gt = (cell_layout == 1).sum()
    pr = (cell_layout == 2).sum() // 2
    bth = (cell_layout == 3).sum() // 3 + 1
    iou = bth / (bth + gt + pr)
    if iou < 0.2:
        return 1 if gt > pr else 2
    else:
        return 3


if __name__ == '__main__':
    main()

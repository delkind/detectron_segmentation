import os

import cv2
import pandas
from shapely.geometry import Polygon
from tqdm import tqdm

from localize_brain import detect_brain
from predict_experiment import get_downloaded_experiments, get_all_experiments


def get_cell_mask(exp_dir, prefix, hippo_mask):
    cell_mask_file_name = os.path.join(exp_dir, prefix + '-cellmask.png')
    cell_mask = cv2.imread(cell_mask_file_name, cv2.IMREAD_GRAYSCALE)
    cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [Polygon(cnt.squeeze()) for cnt in cnts if cnt.shape[0] > 2]
    cnts = [poly for poly in cnts if hippo_mask[int(poly.centroid.y), int(poly.centroid.x)]]
    return cnts


def get_brain_area(exp_dir, prefix):
    thumbnail_file_name = os.path.join(exp_dir, prefix + '-thumb.jpg')
    thumbnail = cv2.imread(thumbnail_file_name, cv2.IMREAD_GRAYSCALE)
    brain_mask, bbox, ctrs = detect_brain(thumbnail)
    brain_area = Polygon(ctrs.squeeze()).area * 64 * 64
    return brain_area


def load_hippo_mask(exp_dir, prefix):
    hippo_mask_file_name = os.path.join(exp_dir, prefix + '-hippomask.png')
    hippo_mask = cv2.imread(hippo_mask_file_name, cv2.IMREAD_GRAYSCALE)
    hippo_ctrs, _ = cv2.findContours(hippo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hippo_mask = cv2.fillPoly(hippo_mask, hippo_ctrs, color=255).astype(bool)
    hippo_ctrs = [Polygon(cnt.squeeze()) for cnt in hippo_ctrs]
    return hippo_ctrs, hippo_mask


def create_section_list(experiment_id, experiments_dir):
    exp_dir = os.path.join(experiments_dir, str(experiment_id))
    sections = [f for f in os.listdir(exp_dir) if os.path.isfile(os.path.join(exp_dir, f))]
    sections = {int(f.split('-')[1]) for f in sections if 'cellmask' in f}
    sections = sorted(list(sections))
    suffixes = ['-cellmask.png', '-hippomask.png', '-thumb.jpg']
    sections = [i for i in sections if
                all([os.path.isfile(os.path.join(experiments_dir, f'{experiment_id}/{experiment_id}-{i}{s}')) for s in
                     suffixes])]
    return sections


def process_section(experiments_dir, experiment_id, section_id):
    exp_dir = os.path.join(experiments_dir, str(experiment_id))
    prefix = f'{experiment_id}-{section_id}'
    hippo_ctrs, hippo_mask = load_hippo_mask(exp_dir, prefix)
    brain_area = get_brain_area(exp_dir, prefix)
    hippo_area = sum([p.area for p in hippo_ctrs])
    cnts = get_cell_mask(exp_dir, prefix, hippo_mask)
    cell_area = sum([poly.area for poly in cnts])
    cell_area_sq = sum([poly.area ** 2 for poly in cnts])
    cell_perimeter = sum([poly.length for poly in cnts])
    cell_perimeter_sq = sum([poly.length ** 2 for poly in cnts])
    return brain_area, hippo_area, len(cnts), cell_area, cell_area_sq, cell_perimeter, cell_perimeter_sq


def main(experiments_dir):
    experiment_fields_to_save = [
        'id',
        'gender',
        'injection_structures',
        'injection_volume',
        'injection_x',
        'injection_y',
        'injection_z',
        'product_id',
        'specimen_name',
        'strain',
        'structure_abbrev',
        'structure_id',
        'structure_name',
        'transgenic_line',
        'transgenic_line_id',
        'primary_injection_structure'
    ]
    experiment_ids = get_downloaded_experiments(experiments_dir)
    experiments, _ = get_all_experiments('output/experiments')
    experiments = {e['id']: e for e in experiments if e['id'] in experiment_ids}
    # image_api = ImageDownloadApi()
    # for i in tqdm(experiment_ids, "Downloading section information"):
    #     sections = image_api.section_image_query(i)
    #     experiments[i]['images'] = {s['section_number']: s for s in sections}
    sections = [(exp, sec) for exp in experiment_ids for sec in create_section_list(exp, experiments_dir)]
    results = []
    for exp, sec in tqdm(sorted(sections), "Processing images: "):
        brain_area, hippo_area, cell_count, cell_area, cell_area_sq, cell_perimeter, cell_perimeter_sq = \
            process_section(experiments_dir, exp, sec)
        result = {
            'brain_area': brain_area,
            'hippo_area': hippo_area,
            'cell_count': cell_count,
            'cell_area': cell_area,
            'cell_area_sq': cell_area_sq,
            'cell_perimeter': cell_perimeter,
            'cell_perimeter_sq': cell_perimeter_sq,
            'section_number': sec,
            'experiment_id': exp,
        }
        result = {**{s: experiments[exp][s] for s in experiment_fields_to_save}, **result}
        results += [result]

    df_list = {key: [res[key] for res in results] for key in results[0]}

    csv = pandas.DataFrame(df_list)
    csv.to_csv('results.csv')


if __name__ == '__main__':
    main('output/experiments')

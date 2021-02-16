import bz2
import os
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from localize_brain import detect_brain
from util import infinite_dict

acronyms = {632: 'DG-sg',
            10703: 'DG-mo',
            382: 'CA1',
            407: 'CA1sp',
            10704: 'DG-po',
            463: 'CA3',
            495: 'CA3sp',
            423: 'CA2',
            446: 'CA2sp',
            403: 'MEA',
            982: 'FC',
            19: 'IG',
            726: 'DG'}


def create_stats(experiments):
    pool = Pool(16)
    return list(tqdm(pool.imap(aggregate_experiment, experiments),
                     "Processing experiments",
                     total=len(experiments)))


def calculate_stats(cells, globs):
    result = infinite_dict()

    result['count'] = len(cells)
    result['count_left'] = len(cells[cells.side == 'left'])
    result['count_right'] = len(cells[cells.side == 'right'])
    for param, value in globs.items():
        result[param] = value
    result['density'] = len(cells) / globs['region_area'] if globs['region_area'] > 0 else 0

    for field in ['coverage', 'area', 'perimeter']:
        result[field] = defaultdict(dict)
        result[field]['mean'] = cells[field].mean() if len(cells) > 0 else 0
        result[field]['median'] = cells[field].median() if len(cells) > 0 else 0

    result['section_count'] = len(cells.section.unique())
    #
    # for section in sections:
    #     cells_section = cells[cells.section == section]
    #     result['sections'][section] = dict()
    #     result['sections'][section]['count'] = len(cells_section)
    #     for field in ['coverage', 'area', 'perimeter']:
    #         result['sections'][section][field] = defaultdict(dict)
    #         result['sections'][section][field]['mean'] = cells_section[field].mean()
    #         result['sections'][section][field]['median'] = cells_section[field].median()

    return result


def calculate_global_parameters(maps, cells, seg_data):
    relevant_sections = cells.section.unique()
    relevant_sections.sort()
    globs_per_section = infinite_dict()
    for section in relevant_sections:
        section_seg_data = seg_data[:, :, section]
        for region, (start, end, shape, (mask_y, mask_x, mask_section)) in maps['dense_masks'].items():
            if start <= section <= end:
                if shape[0] < section_seg_data.shape[0]:
                    ratio = section_seg_data.shape[0] // shape[0]
                    deltas_x = np.array(([i for i in range(ratio)] * ratio) * len(mask_x))
                    deltas_y = np.array([[i] * ratio for i in range(ratio)] * len(mask_x))
                    mask_y = np.kron(mask_y * ratio, np.ones((1, ratio * ratio))).flatten() + deltas_y
                    mask_x = np.kron(mask_x * ratio, np.ones((1, ratio * ratio))).flatten() + deltas_x
                elif shape[0] > section_seg_data.shape[0]:
                    ratio = shape[0] // section_seg_data.shape[0]
                    section_seg_data = np.kron(section_seg_data, np.ones((ratio, ratio), dtype=section_seg_data.dtype))

                relevant_cells = mask_section == (section - start)
                section_seg_data[mask_y[relevant_cells], mask_x[relevant_cells]] = region

        section_density_map = maps['density3d_maps'][2][:, :, section - maps['density3d_maps'][0]]

        if section_density_map.shape[0] > section_seg_data.shape[0]:
            ratio = section_density_map.shape[0] // section_seg_data.shape[0]
            section_seg_data = np.kron(section_seg_data, np.ones((ratio, ratio), dtype=section_seg_data.dtype))
        elif section_density_map.shape[0] < section_seg_data.shape[0]:
            ratio = section_seg_data.shape[0] // section_density_map.shape[0]
            section_density_map = np.kron(section_density_map, np.ones((ratio, ratio), dtype=section_density_map.dtype))

        _, bbox, _ = detect_brain((section_seg_data != 0).astype(np.uint8) * 255)
        center_x = bbox.x + bbox.w // 2
        scale_factor = (0.35 * 64) / (section_seg_data.shape[0] / seg_data.shape[0])
        relevant_regions = np.intersect1d(np.unique(section_seg_data),
                                          cells[cells.section == section].structure_id.unique())
        for region in relevant_regions:
            region_cells = np.where(section_seg_data == region)
            globs_per_section[region][section]['region_area'] = region_cells[0].shape[0] * (scale_factor ** 2)
            globs_per_section[region][section]['region_area_left'] = np.where(region_cells[1] < center_x)[0].shape[
                                                                         0] * (
                                                                             scale_factor ** 2)
            globs_per_section[region][section]['region_area_right'] = np.where(region_cells[1] >= center_x)[0].shape[
                                                                          0] * (
                                                                              scale_factor ** 2)
            densities = section_density_map[region_cells].flatten()
            densities = (densities[densities != 0], len(densities))
            globs_per_section[region][section]['density3d'] = densities

    result = defaultdict(lambda: defaultdict(int))

    for region, region_data in globs_per_section.items():
        section_pairs = list(zip(sorted(region_data.keys())[:-1], sorted(region_data.keys())[1:]))
        if not section_pairs:
            section_pairs = [tuple(region_data.keys()) * 2]

        for s1, s2 in section_pairs:
            volume = (region_data[s1]['region_area'] + region_data[s2]['region_area']) / 2 * 100 * max((s2 - s1), 1)
            density = (region_data[s1]['density3d'][0].sum() + region_data[s2]['density3d'][0].sum()) / (
                    region_data[s1]['density3d'][1] + region_data[s2]['density3d'][1])
            result[region]['count3d'] += volume * density

        for param in ['region_area', 'region_area_left', 'region_area_right']:
            result[region][param] = sum([a[param] for a in region_data.values()])

    return result


def aggregate_experiment(t):
    experiment, data_dir, seg_data_dir = t
    try:
        result = infinite_dict()
        cells = retrieve_celldata(experiment, data_dir)
        maps = pickle.load(bz2.open(os.path.join(f'{data_dir}/{experiment}', f'maps.pickle.bz2'), 'rb'))
        seg_data = np.load(f'{seg_data_dir}/{experiment}/{experiment}-sections.npz')['arr_0']
        globs = calculate_global_parameters(maps, cells, seg_data)

        params = set(list(globs.values())[0].keys())

        struct_sets = {
            382: [382, 391, 399, 407, 415],
            423: [423, 431, 438, 446, 454],
            463: [463, 471, 479, 486, 495, 504],
            726: [726, 10703, 10704, 632, 10702, 734, 742, 751, 758,
                  766, 775, 782, 790, 799, 807, 815, 823]
        }

        structs = {
            **{k: [k] for k in globs.keys()},
            **struct_sets
        }

        for struct, struct_set in structs.items():
            cells_struct = cells[cells.structure_id.isin(struct_set)]
            for dense in cells_struct.dense.unique():
                result[struct]['dense' if dense else 'sparse'] = calculate_stats(
                    cells_struct[cells_struct.dense == dense],
                    {p: sum([globs[s][p] for s in set(globs.keys()).intersection(set(struct_set))]) for p in params})

        result['total'] = calculate_stats(cells, {p: sum([globs[s][p] for s in globs.keys()]) for p in params})
        result = {acronyms.get(k, k): v for k, v in result.items()}
        return experiment, result
    except Exception as e:
        print(f"Exception in experiment {experiment}")
        raise e


def retrieve_celldata(experiment, data_dir):
    celldata_file = os.path.join(f'{data_dir}/{experiment}', f'celldata-{experiment}.parquet')
    return pd.read_parquet(celldata_file)


def perform_aggregation(data_dir, seg_data_dir):
    experiments = [i for i in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{i}')]
    experiments = [(i, data_dir, seg_data_dir) for i in experiments]
    stats_path = f'{data_dir}/../stats.pickle'
    results = create_stats(experiments)
    results = {k: v for k, v in results}
    with open(stats_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <data_dir> <section_data_dir>")
        sys.exit(-1)
    perform_aggregation(sys.argv[1], sys.argv[2])
    # aggregate_experiment((100140949, sys.argv[1], sys.argv[2]))

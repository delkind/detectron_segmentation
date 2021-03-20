import bz2
import os
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from tqdm import tqdm

mcc = MouseConnectivityCache(manifest_file='mouse_connectivity/mouse_connectivity_manifest.json', resolution=25)

if os.path.isfile('mouse_connectivity/tree.pickle'):
    acronyms, structs_descendants, structs_children = pickle.load(open('mouse_connectivity/tree.pickle', 'rb'))
else:
    acronyms = {v: k for k, v in mcc.get_structure_tree().get_id_acronym_map().items()}
    structs_descendants = {i: set(mcc.get_structure_tree().descendant_ids([i])[0])
                           for i in set(mcc.get_structure_tree().descendant_ids([8])[0])}
    structs_children = {i: set([a['id'] for a in mcc.get_structure_tree().children([i])[0]])
                        for i in structs_descendants.keys()}
    pickle.dump((acronyms, structs_descendants, structs_children), open('mouse_connectivity/tree.pickle', 'wb'))

conversion = {
    'volume': -3,
    'region_area_right': -2,
    'region_area': -2,
    'region_area_left': -2,
    'area': -2,
    'perimeter': -1,
}


def create_stats(experiments):
    pool = Pool(64)
    return list(tqdm(pool.imap(process_experiment, experiments),
                     "Processing experiments",
                     total=len(experiments)))
    # return list(tqdm(map(process_experiment, experiments),
    #                  "Processing experiments",
    #                  total=len(experiments)))


def calculate_stats(cells, globs, struct, sections, seg):
    params = set(list(globs.values())[0].keys())
    result = {p: sum([globs[s][p] for s in set(globs.keys()).intersection(set(cells.structure_id.unique().tolist()))])
              for p in params}
    result = {
        **{k: v * 10.0 ** (3 * conversion.get(k, 0)) for k, v in result.items()},
        **{field: {
            'mean': cells[field].mean() * 10.0 ** (3 * conversion.get(field, 0)) if len(cells) > 0 else 0,
            'median': cells[field].median() * 10.0 ** (3 * conversion.get(field, 0)) if len(cells) > 0 else 0,
            'percentile10': cells[field].quantile(0.1) * 10.0 ** (3 * conversion.get(field, 0)) if len(
                cells) > 0 else 0,
            'percentile90': cells[field].quantile(0.9) * 10.0 ** (3 * conversion.get(field, 0)) if len(
                cells) > 0 else 0,
            'percentile95': cells[field].quantile(0.95) * 10.0 ** (3 * conversion.get(field, 0)) if len(
                cells) > 0 else 0,
        } for field in ['coverage', 'area', 'perimeter']},
    }

    struct_map = np.isin(seg, list(struct))
    brightnesses = [[d[struct_map[:d.shape[0], :d.shape[1], section]] for d in data]
                    for section, data in sections.items()]
    brightness, injection = tuple(zip(*brightnesses))

    brightness = np.concatenate(brightness)
    injection = np.concatenate(injection)

    if len(brightness) == 0:
        brightness = np.array([0, 0])
    if len(injection) == 0:
        injection = np.array([0, 0])

    result = {**result,
              'count': len(cells), 'count_left': len(cells[cells.side == 'left']),
              'count_right': len(cells[cells.side == 'right']), 'section_count': len(cells.section.unique()),
              'density': len(cells) / result['region_area'] if result['region_area'] > 0 else 0,
              'density3d': result['count3d'] / result['volume'] if result['volume'] > 0 else 0,
              'brightness': {'mean': np.mean(brightness), 'median': np.median(brightness),
                             'percentile90': np.percentile(brightness, 90)},
              'injection': {'mean': np.mean(injection), 'median': np.median(injection),
                            'percentile90': np.percentile(injection, 90)},
              }

    return result


def calculate_global_parameters(globs_per_section):
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
            result[region]['volume'] += volume

        for param in ['region_area', 'region_area_left', 'region_area_right']:
            result[region][param] = sum([a[param] for a in region_data.values()])

    return result


def get_struct_aggregates(relevant_structs):
    relevant_aggregates = {k: relevant_structs.intersection(v)
                           for k, v in structs_descendants.items() if relevant_structs.intersection(v)}
    return relevant_aggregates


def process_experiment(t):
    experiment, data_dir, struct_data_dir = t
    try:
        cells = retrieve_celldata(experiment, data_dir)
        seg = np.load(f'{struct_data_dir}/{experiment}/{experiment}-sections.npz')['arr_0']

        with open(f'{data_dir}/{experiment}/bboxes.pickle', 'rb') as f:
            bboxes = pickle.load(f)
        sections = sorted([s for s in bboxes.keys() if bboxes[s]])
        sections = {s: (cv2.imread(f"{data_dir}/{experiment}/thumbnail-{experiment}-{s}.jpg",
                                   cv2.IMREAD_COLOR)[:seg.shape[0], :seg.shape[1], 1],
                        cv2.imread(f"{data_dir}/{experiment}/thumbnail-{experiment}-{s}.jpg",
                                   cv2.IMREAD_GRAYSCALE)[:seg.shape[0], :seg.shape[1]]) for s in sections}

        maps = pickle.load(bz2.open(os.path.join(f'{data_dir}/{experiment}', f'maps.pickle.bz2'), 'rb'))
        globs = calculate_global_parameters(maps['globs'])

        relevant_structs = set(globs.keys())
        relevant_aggregates = get_struct_aggregates(relevant_structs)

        reverse_structs = defaultdict(list)
        for k, v in relevant_aggregates.items():
            reverse_structs[tuple(sorted(v))].append(k)

        brightness_data = dict()
        for struct in reverse_structs.keys():
            brightness_data[struct] = {}

        aggregate_data = {struct_set: calculate_stats(cells[cells.structure_id.isin(struct_set)], globs, struct_set,
                                                      sections, seg) for struct_set in reverse_structs.keys()}
        result = {acronyms[k]: {**data, **brightness_data.get(k, {'brightness': 0, 'injection': 0})}
                  for s, data in aggregate_data.items() for k in reverse_structs[s]}

        return experiment, result
    except Exception as e:
        print(f"Exception in experiment {experiment}")
        raise e


def retrieve_celldata(experiment, data_dir):
    celldata_file = os.path.join(f'{data_dir}/{experiment}', f'celldata-{experiment}.parquet')
    return pd.read_parquet(celldata_file)


def perform_aggregation(data_dir, struct_data_dir):
    experiments = [i for i in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{i}')]
    # experiments = ['100140949']
    experiments = [(i, data_dir, struct_data_dir) for i in experiments]
    stats_path = f'{data_dir}/../stats.pickle'
    results = create_stats(experiments)

    common_structs = set.intersection(*[set(v.keys()) for k, v in results])
    results = {k: {s: d for s, d in v.items() if s in common_structs} for k, v in results}
    with open(stats_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <data_dir> <struct_data_dir>")
        sys.exit(-1)
    perform_aggregation(sys.argv[1], sys.argv[2])

import bz2
import functools
import os
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from util import infinite_dict

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

mcc = MouseConnectivityCache(manifest_file='mouse_connectivity/mouse_connectivity_manifest.json', resolution=25)

acronyms = {v: k for k, v in mcc.get_structure_tree().get_id_acronym_map().items()}


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

        result[region]['density3d'] = result[region]['count3d'] / result[region]['volume']

        for param in ['region_area', 'region_area_left', 'region_area_right']:
            result[region][param] = sum([a[param] for a in region_data.values()])

    return result


def aggregate_experiment(t):
    experiment, data_dir = t
    try:
        result = infinite_dict()
        cells = retrieve_celldata(experiment, data_dir)
        maps = pickle.load(bz2.open(os.path.join(f'{data_dir}/{experiment}', f'maps.pickle.bz2'), 'rb'))
        globs = calculate_global_parameters(maps['globs'])

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
            dense_options = cells_struct.dense.unique()
            for dense in dense_options:
                res = calculate_stats(cells_struct[cells_struct.dense == dense],
                                        {p: sum([globs[s][p] for s in set(globs.keys()).intersection(set(struct_set))])
                                         for p in params})
                if len(dense_options) > 1:
                    result[struct]['dense' if dense else 'sparse'] = res
                else:
                    result[struct] = res

        result['total'] = calculate_stats(cells, {p: sum([globs[s][p] for s in globs.keys()]) for p in params})
        result = {acronyms.get(k, k): v for k, v in result.items()}
        return experiment, result
    except Exception as e:
        print(f"Exception in experiment {experiment}")
        raise e


def retrieve_celldata(experiment, data_dir):
    celldata_file = os.path.join(f'{data_dir}/{experiment}', f'celldata-{experiment}.parquet')
    return pd.read_parquet(celldata_file)


def perform_aggregation(data_dir):
    experiments = [i for i in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{i}')]
    experiments = [(i, data_dir) for i in experiments]
    stats_path = f'{data_dir}/../stats.pickle'
    results = create_stats(experiments)

    common_structs = set.intersection(*[set(v.keys()) for k, v in results])
    results = {k: {s: d for s, d in v.items() if s in common_structs} for k, v in results}
    with open(stats_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <data_dir>")
        sys.exit(-1)
    perform_aggregation(sys.argv[1])
    # aggregate_experiment((100140949, sys.argv[1]))

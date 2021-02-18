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


def calculate_stats(cells, globs):
    params = set(list(globs.values())[0].keys())
    result = {p: sum([globs[s][p] for s in set(globs.keys()).intersection(set(cells.structure_id.unique().tolist()))])
              for p in params}
    result = {
        **{k: v * 10.0 ** (3 * conversion.get(k, 0)) for k, v in result.items()},
        **{field: {
            'mean': cells[field].mean() * 10.0 ** (3 * conversion.get(field, 0)) if len(cells) > 0 else 0,
            'median': cells[field].median() * 10.0 ** (3 * conversion.get(field, 0)) if len(cells) > 0 else 0
        } for field in ['coverage', 'area', 'perimeter']},
    }

    result = {**result,
              'count': len(cells), 'count_left': len(cells[cells.side == 'left']),
              'count_right': len(cells[cells.side == 'right']), 'section_count': len(cells.section.unique()),
              'density': len(cells) / result['region_area'] if result['region_area'] > 0 else 0,
              'density3d': result['count3d'] / result['volume'] if result['volume'] > 0 else 0
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


def process_experiment(t):
    experiment, data_dir = t
    try:
        cells = retrieve_celldata(experiment, data_dir)
        maps = pickle.load(bz2.open(os.path.join(f'{data_dir}/{experiment}', f'maps.pickle.bz2'), 'rb'))
        globs = calculate_global_parameters(maps['globs'])

        structs = {
            382: [382, 391, 399, 407, 415],
            423: [423, 431, 438, 446, 454],
            463: [463, 471, 479, 486, 495, 504],
            726: [726, 10703, 10704, 632, 10702, 734, 742, 751, 758,
                  766, 775, 782, 790, 799, 807, 815, 823],
            **{p: [p] for p in globs.keys()}
        }

        result = {struct: calculate_stats(cells[cells.structure_id.isin(struct_set)], globs)
                  for struct, struct_set in structs.items()}
        result = {acronyms[k]: v for k, v in result.items()}
        result['total'] = calculate_stats(cells, globs)
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
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data_dir>")
        sys.exit(-1)
    perform_aggregation(sys.argv[1])
    # process_experiment((100140949, sys.argv[1]))

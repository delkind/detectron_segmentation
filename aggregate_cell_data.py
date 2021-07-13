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
    if len(experiments) == 1:
        process_experiment(experiments[0])
    else:
        pool = Pool(48)
        return list(tqdm(pool.imap(process_experiment, experiments),
                         "Processing experiments",
                         total=len(experiments)))


def calculate_stats(cells, globs, structs, sections, seg):
    unique_sections = sorted(cells.section.unique().tolist())
    structs = [d for s in structs for d in structs_descendants[s]]

    struct_map = np.isin(seg, list(structs))
    brightnesses = [[d[struct_map[:d.shape[0], :d.shape[1], section]] for d in data]
                    for section, data in sections.items()]
    brightness, injection = tuple(zip(*brightnesses))

    brightness = np.concatenate(brightness)
    injection = np.concatenate(injection)

    if len(brightness) == 0:
        brightness = np.array([0, 0])
    if len(injection) == 0:
        injection = np.array([0, 0])

    params = set(list(globs.values())[0]['all'].keys())
    gl = {
        p: sum([globs[s]['all'][p] for s in set(globs.keys()).intersection(set(cells.structure_id.unique().tolist()))])
        for p in params}
    gl = {
        **{k: v * 10.0 ** (3 * conversion.get(k, 0)) for k, v in gl.items()},
    }

    result = {**gl,
              'section_count': len(unique_sections),
              'density_left':
                  len(cells[cells.side == 'left']) / gl['region_area_left']
                  if gl['region_area_left'] > 0 else 0,
              'density_right':
                  len(cells[cells.side == 'right']) / gl['region_area_right']
                  if gl['region_area_right'] > 0 else 0,
              'density3d': gl['count3d'] / gl['volume'] if gl['volume'] > 0 else 0,
              'brightness': {'mean': np.mean(brightness), 'median': np.median(brightness),
                             'percentile90': np.percentile(brightness, 90)},
              'injection': {'mean': np.mean(injection), 'median': np.median(injection),
                            'percentile90': np.percentile(injection, 90)},
              'density': len(cells) / gl['region_area'] if gl['region_area'] > 0 else 0,
              **calculate_section_dependent_data(cells, globs),
              **{'sections': {section: calculate_section_dependent_data(cells[cells.section == section], globs)
                              for section in unique_sections}}
              }

    return result


def calculate_section_dependent_data(cells, globs):
    sections = cells.section.unique().tolist()
    if len(sections) == 1:
        area = sum([globs[s][sections[0]]['region_area'] * 10.0 ** (3 * conversion.get('region_area', 0)) for s in cells.structure_id.unique()])
        density = len(cells) / area if area != 0 else 0
        appendix = {'region_area': area, 'density': density}
    else:
        appendix = dict()

    return {**{field: {
        'mean': cells[field].mean() * 10.0 ** (3 * conversion.get(field, 0)) if len(cells) > 0 else 0,
        'median': cells[field].median() * 10.0 ** (3 * conversion.get(field, 0)) if len(cells) > 0 else 0,
        **{f'percentile{le}': cells[field].quantile(le / 100) * 10.0 ** (3 * conversion.get(field, 0)) if len(
            cells) > 0 else 0 for le in [1, 5, 10, 90, 95, 99]},
    } for field in ['coverage', 'area', 'perimeter', 'diameter']},
            'count': len(cells),
            'count_left': len(cells[cells.side == 'left']),
            'count_right': len(cells[cells.side == 'right']),
            **appendix
            }


def calculate_global_parameters(globs_per_section):
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for region, region_data in globs_per_section.items():
        section_pairs = list(zip(sorted(region_data.keys())[:-1], sorted(region_data.keys())[1:]))
        if not section_pairs:
            section_pairs = [tuple(region_data.keys()) * 2]

        for s1, s2 in section_pairs:
            volume = (region_data[s1]['region_area'] + region_data[s2]['region_area']) / 2 * 100 * max((s2 - s1), 1)
            density = (region_data[s1]['density3d'][0].sum() + region_data[s2]['density3d'][0].sum()) / (
                    region_data[s1]['density3d'][1] + region_data[s2]['density3d'][1])
            result[region]['all']['count3d'] += volume * density
            result[region]['all']['volume'] += volume

        for param in ['region_area', 'region_area_left', 'region_area_right']:
            result[region]['all'][param] = sum([a[param] for a in region_data.values()])

        for section in region_data.keys():
            result[region][section] = region_data[section]

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

        aggregate_data = {struct_set: calculate_stats(cells[cells.structure_id.isin(struct_set)], globs, structs,
                                                      sections, seg) for struct_set, structs in reverse_structs.items()}
        result = {acronyms[k]: data for s, data in aggregate_data.items() for k in reverse_structs[s]}

        return experiment, result
    except Exception as e:
        print(f"Exception in experiment {experiment}")
        raise e


def retrieve_celldata(experiment, data_dir):
    celldata_file = os.path.join(f'{data_dir}/{experiment}', f'celldata-{experiment}.parquet')
    return pd.read_parquet(celldata_file)


def perform_aggregation(data_dir, struct_data_dir):
    experiments = [i for i in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{i}')]
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

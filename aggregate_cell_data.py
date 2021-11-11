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
        return [process_experiment(experiments[0])]
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
              'density': len(cells) / gl['region_area'] if gl['region_area'] > 0 else 0,
              'density3d_left': gl['count3d_left'] / gl['volume_left'] if gl['volume_left'] > 0 else 0,
              'density3d_right': gl['count3d_right'] / gl['volume_right'] if gl['volume_right'] > 0 else 0,
              'density3d': gl['count3d'] / gl['volume'] if gl['volume'] > 0 else 0,
              'brightness': {'mean': np.mean(brightness), 'median': np.median(brightness),
                             'percentile90': np.percentile(brightness, 90)},
              'injection': {'mean': np.mean(injection), 'median': np.median(injection),
                            'percentile90': np.percentile(injection, 90)},
              **calculate_section_dependent_data(cells, globs),
              }

    section_data = {section: calculate_section_dependent_data(cells[cells.section == section], globs)
                    for section in unique_sections}

    return result, section_data


def calculate_section_dependent_data(cells, globs):
    sections = cells.section.unique().tolist()
    if len(sections) == 1:
        area = sum([globs[s][sections[0]]['region_area'] * 10.0 ** (3 * conversion.get('region_area', 0)) for s in
                    cells.structure_id.unique()])
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
            volume_left = (region_data[s1]['region_area_left'] + region_data[s2]['region_area_left']) / 2 * 100 * max(abs(s2 - s1), 1)
            volume_right = (region_data[s1]['region_area_right'] + region_data[s2]['region_area_right']) / 2 * 100 * max(abs(s2 - s1), 1)
            density3d_left_sum = region_data[s1]['density3d_left'][0].sum() + region_data[s2]['density3d_left'][0].sum()
            density3d_left_len = region_data[s1]['density3d_left'][1] + region_data[s2]['density3d_left'][1]
            density3d_right_sum = region_data[s1]['density3d_right'][0].sum() + region_data[s2]['density3d_right'][0].sum()
            density3d_right_len = region_data[s1]['density3d_right'][1] + region_data[s2]['density3d_right'][1]

            density = (density3d_left_sum + density3d_right_sum) / (density3d_left_len + density3d_right_len) if (density3d_left_len + density3d_right_len) != 0 else 0
            density_left = density3d_left_sum / density3d_left_len if density3d_left_len != 0 else 0
            density_right = density3d_right_sum / density3d_right_len if density3d_right_len != 0 else 0

            result[region]['all']['count3d_left'] += volume_left * density_left
            result[region]['all']['count3d_right'] += volume_right * density_right
            result[region]['all']['count3d'] += volume * density
            result[region]['all']['volume_left'] += volume_left
            result[region]['all']['volume_right'] += volume_right
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
        result = {acronyms[k]: data[0] for s, data in aggregate_data.items() for k in reverse_structs[s]}
        section_data = {acronyms[k]: data[1] for s, data in aggregate_data.items() for k in reverse_structs[s]}

        return experiment, result, section_data
    except Exception as e:
        print(f"Exception in experiment {experiment}")
        raise e


def process_experiment_section_data(experiment, experiment_data):
    data = list()
    for region, region_data in experiment_data.items():
        d = defaultdict(list)
        for section, section_data in region_data.items():
            for param, param_data in section_data.items():
                if isinstance(param_data, dict):
                    d[param].append(param_data['mean'])
                else:
                    d[param].append(param_data)
        reg_dict = dict()
        for col in d.keys():
            for i, val in enumerate(map(lambda x: x.mean() if x.shape[0] > 0 else 0, np.array_split(d[col], 10))):
                reg_dict[f'{col}_{i}'] = val
        reg_dict['experiment_id'] = experiment
        reg_dict['region'] = region
        data.append(reg_dict)
    return data


def retrieve_celldata(experiment, data_dir):
    celldata_file = os.path.join(f'{data_dir}/{experiment}', f'celldata-{experiment}.parquet')
    return pd.read_parquet(celldata_file)


def build_region_row(exp_id, region, reg_data):
    params_dict = dict()
    for param, param_data in reg_data.items():
        if type(param_data) != dict:
            params_dict[param] = param_data
        else:
            params_dict = {**params_dict, **{f'{param}|{stat}': stat_val for stat, stat_val in param_data.items()}}

    params_dict = {
        'experiment_id': int(exp_id),
        'region': region,
        **params_dict
    }

    return params_dict


def perform_aggregation(data_dir, struct_data_dir):
    experiments = [i for i in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{i}')]
    experiments = [(i, data_dir, struct_data_dir) for i in experiments]
    stats_path = f'{data_dir}/../stats.parquet'
    results_list = create_stats(experiments)
    section_data_path = f'{data_dir}/../stats-sections.parquet'

    common_structs = set.intersection(*[set(v.keys()) for k, v, _ in results_list])
    results = {int(k): {s: d for s, d in v.items() if s in common_structs} for k, v, _ in results_list}
    data_rows = [build_region_row(exp_id, region, reg_data) for exp_id, exp_data in results.items()
                 for region, reg_data in exp_data.items()]
    data_frame = pd.DataFrame(data_rows)
    data_frame.sort_values(['experiment_id', 'region']).to_parquet(stats_path)

    section_data = {int(k): {s: d for s, d in v.items() if s in common_structs} for k, _, v in results_list}
    section_data_list = list()
    for exp, data in section_data.items():
        section_data_list += process_experiment_section_data(exp, data)
    pd.DataFrame(section_data_list).sort_values(['experiment_id', 'region']).to_parquet(section_data_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <data_dir> <struct_data_dir>")
        sys.exit(-1)
    perform_aggregation(sys.argv[1], sys.argv[2])

import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

data_dir = 'output/hippo_exp/analyzed/'


def open_all(experiments):
    results = {e: retrieve_celldata(e) for e in tqdm(experiments)}
    return results


def create_stats(experiments):
    pool = Pool(16)
    results = list(tqdm(pool.imap(process_experiment, experiments),
                        "Processing experiments",
                        total=len(experiments)))
    results = {k: v for k, v in results}
    return results


def calculate_stats(cells, sections):
    result = dict()

    result['count'] = len(cells)
    for field in ['density', 'area']:
        result[field] = defaultdict(dict)
        result[field]['mean'] = cells[field].mean()
        result[field]['median'] = cells[field].median()

    result['sections'] = dict()

    for section in sections:
        cells_section = cells[cells.section == section]
        result['sections'][section] = dict()
        result['sections'][section]['count'] = len(cells_section)
        for field in ['density', 'area']:
            result['sections'][section][field] = defaultdict(dict)
            result['sections'][section][field]['mean'] = cells_section[field].mean()
            result['sections'][section][field]['median'] = cells_section[field].median()

    return result


def process_experiment(experiment):
    result = dict()
    cells = retrieve_celldata(experiment)
    sections = sorted(cells.section.unique().tolist())
    result['total'] = calculate_stats(cells, sections)

    result['dense'] = dict()
    result['sparse'] = dict()
    result['region'] = dict()
    for i, struct in enumerate([382, 423, 463]):
        result['dense'][f'CA{i + 1}'] = calculate_stats(cells[(cells.structure_id == struct) & cells.dense], sections)
        result['sparse'][f'CA{i + 1}'] = calculate_stats(cells[(cells.structure_id == struct) & (cells.dense == False)],
                                                         sections)
        result['region'][f'CA{i + 1}'] = calculate_stats(cells[(cells.structure_id == struct)], sections)

    result['dense'][f'DG'] = calculate_stats(cells[cells.structure_id == 632], sections)
    result['sparse']['DG'] = calculate_stats(cells[cells.structure_id.isin([10703, 10704])], sections)
    result['region']['DG'] = calculate_stats(cells[cells.structure_id.isin([10703, 10704, 632])], sections)
    result['special'] = defaultdict(dict)
    result['special']['DG-mo'] = calculate_stats(cells[cells.structure_id == 10703], sections)
    result['special']['DG-po'] = calculate_stats(cells[cells.structure_id == 10704], sections)

    return experiment, result


def retrieve_celldata(experiment):
    celldata_file = os.path.join(f'{data_dir}/{experiment}', f'celldata-{experiment}.csv')
    return pd.read_csv(celldata_file)


def perform_conversion():
    experiments = [i for i in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{i}')]
    stats_path = f'{data_dir}/../stats.pickle'
    results = create_stats(experiments)
    with open(stats_path, 'wb') as f:
        pickle.dump(results, f)


perform_conversion()

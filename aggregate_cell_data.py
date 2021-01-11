import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from util import infinite_dict

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


def calculate_stats(cells, area):
    result = infinite_dict()

    result['count'] = len(cells)
    result['total_area'] = area
    result['density'] = len(cells) / area
    for field in ['coverage', 'area', 'perimeter']:
        result[field] = defaultdict(dict)
        result[field]['mean'] = cells[field].mean()
        result[field]['median'] = cells[field].median()

    # result['sections'] = dict()
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


def process_experiment(experiment):
    result = infinite_dict()
    cells = retrieve_celldata(experiment)
    areas = pickle.load(open(os.path.join(f'{data_dir}/{experiment}', f'areas.pickle'), 'rb'))
    sections = sorted(cells.section.unique().tolist())
    result['total'] = calculate_stats(cells, sum([sum([areas[s][d] for s in areas.keys()])
                                              for d in ['sparse', 'dense']]))
    result['total']['section_count'] = len(sections)

    for i, struct in enumerate([382, 423, 463]):
        result['dense'][f'CA{i + 1}'] = calculate_stats(cells[(cells.structure_id == struct) & cells.dense],
                                                        areas[struct]['dense'])
        result['sparse'][f'CA{i + 1}'] = calculate_stats(cells[(cells.structure_id == struct) & (cells.dense == False)],
                                                         areas[struct]['sparse'])
        result['region'][f'CA{i + 1}'] = calculate_stats(cells[(cells.structure_id == struct)],
                                                         areas[struct]['dense'] + areas[struct]['sparse'])

    result['dense'][f'DG'] = calculate_stats(cells[cells.structure_id == 632], areas[632]['dense'])
    result['sparse']['DG'] = calculate_stats(cells[cells.structure_id.isin([10703, 10704])], areas[632]['sparse'])
    result['region']['DG'] = calculate_stats(cells[cells.structure_id.isin([10703, 10704, 632])],
                                             areas[632]['sparse'] + areas[632]['dense'])

    return experiment, result


def retrieve_celldata(experiment):
    celldata_file = os.path.join(f'{data_dir}/{experiment}', f'celldata-{experiment}.parquet')
    return pd.read_parquet(celldata_file)


def perform_aggregation():
    experiments = [i for i in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{i}')]
    stats_path = f'{data_dir}/../stats.pickle'
    results = create_stats(experiments)
    with open(stats_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    perform_aggregation()
import os
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from util import infinite_dict


def create_stats(experiments):
    pool = Pool(16)
    results = list(tqdm(pool.imap(aggregate_experiment, experiments),
                        "Processing experiments",
                        total=len(experiments)))
    results = {k: v for k, v in results}
    return results


def calculate_stats(cells, area):
    result = infinite_dict()

    result['count'] = len(cells)
    result['total_area'] = area
    result['density'] = len(cells) / area if area > 0 else 0
    for field in ['coverage', 'area', 'perimeter']:
        result[field] = defaultdict(dict)
        result[field]['mean'] = cells[field].mean() if len(cells) > 0 else 0
        result[field]['median'] = cells[field].median() if len(cells) > 0 else 0

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


def aggregate_experiment(experiment, data_dir):
    result = infinite_dict()
    cells = retrieve_celldata(experiment, data_dir)
    areas = pickle.load(open(os.path.join(f'{data_dir}/{experiment}', f'areas.pickle'), 'rb'))
    sections = sorted(cells[cells.structure_id != 403].section.unique().tolist())
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

    result['dense']['DG'] = calculate_stats(cells[cells.structure_id == 632], areas[632]['dense'])
    result['sparse']['DG'] = calculate_stats(cells[cells.structure_id.isin([10703, 10704])], areas[632]['sparse'])
    result['region']['DG'] = calculate_stats(cells[cells.structure_id.isin([10703, 10704, 632])],
                                             areas[632]['sparse'] + areas[632]['dense'])

    cells_mea = cells[(cells.structure_id == 403)]
    result['dense']['MEA'] = calculate_stats(cells_mea[cells_mea.dense], areas[403]['dense'])
    result['sparse']['MEA'] = calculate_stats(cells_mea[cells_mea.dense == False], areas[403]['sparse'])
    result['region']['MEA'] = calculate_stats(cells_mea, areas[403]['sparse'] + areas[403]['dense'])

    return experiment, result


def retrieve_celldata(experiment, data_dir):
    celldata_file = os.path.join(f'{data_dir}/{experiment}', f'celldata-{experiment}.parquet')
    return pd.read_parquet(celldata_file)


def perform_aggregation(data_dir):
    experiments = [i for i in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{i}')]
    stats_path = f'{data_dir}/../stats.pickle'
    results = create_stats(experiments)
    with open(stats_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data_dir>")
        sys.exit(-1)
    perform_aggregation(sys.argv[1])

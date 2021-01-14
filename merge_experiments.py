import os
import pickle
import shutil
import sys
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm


def merge_boxes(main_dir, source_dir):
    bboxes_main = pickle.load(open(f'{main_dir}/bboxes.pickle', 'rb'))
    bboxes_source = pickle.load(open(f'{source_dir}/bboxes.pickle', 'rb'))
    bboxes_merged = dict()

    sections = sorted([k for k, v in bboxes_main.items() if v] + [k for k, v in bboxes_source.items() if v])

    for section in sections:
        bboxes_merged[section] = bboxes_main.get(section, []) + bboxes_source.get(section, [])

    pickle.dump(bboxes_merged, open(f'{main_dir}/bboxes.pickle', 'wb'))


def merge_areas(main_dir, source_dir):
    areas_main = pickle.load(open(f'{main_dir}/areas.pickle', 'rb'))
    areas_source = pickle.load(open(f'{source_dir}/areas.pickle', 'rb'))

    areas_merged = {**areas_main, **areas_source}

    pickle.dump(areas_merged, open(f'{main_dir}/areas.pickle', 'wb'))


def merge_celldata(main_dir, source_dir, experiment_id):
    celldata_main = pd.read_parquet(f'{main_dir}/celldata-{experiment_id}.parquet')
    celldata_source = pd.read_parquet(f'{source_dir}/celldata-{experiment_id}.parquet')

    celldata_merged = pd.concat([celldata_main, celldata_source])

    celldata_merged.to_parquet(f'{main_dir}/celldata-{experiment_id}.parquet')


def merge_experiment(t):
    main_dir, source_dir, experiment_id = t
    merge_boxes(main_dir, source_dir)
    merge_areas(main_dir, source_dir)
    merge_celldata(main_dir, source_dir, experiment_id)

    images_to_copy = os.listdir(source_dir)
    images_in_dest = os.listdir(main_dir)

    for f in images_to_copy:
        if f not in images_in_dest and any([f.startswith(p) for p in ['full', 'cellmask', 'thumbnail']]):
            print(f'Copying {f}...')
            shutil.copy(f'{source_dir}/{f}', f'{main_dir}/{f}')


def merge_experiments(main_dir, source_dir):
    main_experiments = os.listdir(main_dir)
    experiments = [(f'{main_dir}/{i}', f'{source_dir}/{i}', i) for i in os.listdir(source_dir)
                   if os.path.isdir(f'{source_dir}/{i}') and i in main_experiments]

    pool = Pool(16)
    results = list(tqdm(pool.imap(merge_experiment, experiments),
                        "Processing experiments",
                        total=len(experiments)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <main_dir> <source_dir>")
        sys.exit(-1)
    merge_experiments(sys.argv[1], sys.argv[2])

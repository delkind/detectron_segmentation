import os
import pickle
import shutil

import pandas as pd


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


def merge_experiment(main_dir, source_dir, experiment_id):
    merge_boxes(main_dir, source_dir)
    merge_areas(main_dir, source_dir)
    merge_celldata(main_dir, source_dir, experiment_id)

    images_to_copy = os.listdir(source_dir)
    images_in_dest = os.listdir(main_dir)

    for f in images_to_copy:
        if f not in images_in_dest and any([f.startswith(p) for p in ['full', 'cellmask', 'thumbnail']]):
            print(f'Copying {f}...')
            shutil.copy(f'{source_dir}/{f}', f'{main_dir}/{f}')


merge_experiment('./output/hippo_exp/analyzed/100140949', './output/amygdala/analyzed/100140949', 100140949)
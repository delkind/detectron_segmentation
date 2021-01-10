import pandas
from tqdm import tqdm

from convert_masks import get_downloaded_experiments


def get_all_experiments(connectivity_dir):
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
    mcc = MouseConnectivityCache(manifest_file=f'{connectivity_dir}/mouse_connectivity_manifest.json',
                                 resolution=MouseConnectivityApi.VOXEL_RESOLUTION_100_MICRONS)
    experiments = mcc.get_experiments(dataframe=False)
    return experiments, mcc


def main(experiments_dir):
    experiment_fields_to_save = [
        'id',
        'gender',
        'primary_injection_structure',
        'strain',
        'transgenic_line',
        'structure_name',
        'specimen_name',
        'transgenic_line_id',
        'injection_structures',
        'injection_volume',
        'injection_x',
        'injection_y',
        'injection_z',
        'product_id',
    ]
    experiment_ids = get_downloaded_experiments(experiments_dir)
    experiments, _ = get_all_experiments('./mouse_connectivity')
    experiments = {e['id']: e for e in experiments if e['id'] in experiment_ids}
    results = []
    for exp in tqdm(experiment_ids, "Downloading section information"):
        result = {s: experiments[exp][s] for s in experiment_fields_to_save}
        results += [result]

    df_list = {key: [res[key] for res in results] for key in results[0]}
    csv = pandas.DataFrame(df_list)
    csv.to_csv('results.csv')


if __name__ == '__main__':
    main('output/hippo_exp/analyzed')

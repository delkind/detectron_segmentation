import argparse
import os
import shutil
import subprocess

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


def main(output_dir, resolution, retain_transform_data, zoom, parallel_downloads, parallel_processors):
    mcc = MouseConnectivityCache(manifest_file=f'{output_dir}/connectivity/mouse_connectivity_manifest.json',
                                 resolution=resolution)
    mcc.get_annotation_volume()
    mcc.get_reference_space()
    experiments = [e['id'] for e in mcc.get_experiments(dataframe=False)]
    experiments = sorted(list(set(experiments)))

    try:
        os.makedirs(f'{output_dir}/data/input')
        for e in experiments:
            os.makedirs(f'{output_dir}/data/input/{e}')
    except FileExistsError:
        for e in experiments:
            if os.path.isdir(f'{output_dir}/data/input/{e}'):
                shutil.rmtree(f'{output_dir}/data/input/{e}')
                os.makedirs(f'{output_dir}/data/input/{e}')

    downloads = [subprocess.Popen(["python",
                                   "./displacement_data_downloader.py",
                                   f"-o{output_dir}",
                                   f"-r{resolution}",
                                   f"-n{i}"])
                 for i in range(parallel_downloads)]
    processes = [subprocess.Popen(["python",
                                   "./segmentation_data_builder.py",
                                   f"-o{output_dir}",
                                   f"-r{resolution}",
                                   f"-c{len(experiments)}",
                                   f"-z{zoom}"
                                   ] +
                                  (["-t"] if retain_transform_data else []) +
                                  [f"-n{i}"])
                 for i in range(parallel_processors)]
    exit_codes = [p.wait() for p in processes + downloads]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build segmentation data for Mouse Connectivity')
    parser.add_argument('--output_dir', '-o', required=True, action='store', help='Directory that will contain output')
    parser.add_argument('--resolution', '-r', default=25, type=int, action='store', help='Reference space resolution')
    parser.add_argument('--zoom', '-z', default=2, type=int, action='store', help='Image zoom')
    parser.add_argument('--retain_transform_data', '-t', action='store_true', default=False,
                        help='Retain the transform data')
    parser.add_argument('--parallel_downloads', '-d', action='store', type=int, required=True,
                        help='Number of parallel downloads')
    parser.add_argument('--parallel_processors', '-p', action='store', type=int, required=True,
                        help='Number of parallel processors')
    args = parser.parse_args()

    print(vars(args))
    main(**vars(args))

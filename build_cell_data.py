import argparse
import os
import re
import subprocess

import pandas


def main(input_dir, output_dir, brain_seg_data_dir, structure_id, parallel_processors, verify_thumbnail):
    items = sorted([item for item in os.listdir(input_dir)
                    if os.path.isdir(os.path.join(input_dir, item))
                    and bool(re.match('^\\d+$', item))])

    if not items:
        return

    try:
        os.makedirs(f'{output_dir}/input')
        for i in items:
            os.makedirs(f'{output_dir}/input/{i}')
    except FileExistsError:
        pass
    processes = [subprocess.Popen(["python",
                                   "./process_cell_data.py",
                                   f"-i{input_dir}",
                                   f"-o{output_dir}",
                                   f"-b{brain_seg_data_dir}",
                                   f"-s{structure_id}",
                                   f"-n{i}",
                                   ] +
                                  (['-t'] if verify_thumbnail else []))
                 for i in range(parallel_processors)]
    exit_codes = [p.wait() for p in processes]

    df = None

    for item in items:
        csv = pandas.read_csv(f'{output_dir}/result/{item}/experiment_data.csv')
        if df is None:
            df = csv
        else:
            df.join(csv)

    df.to_csv(f'{output_dir}/result/experiments_data.csv')


def create_cell_build_argparser():
    parser = argparse.ArgumentParser(description='Build Mouse Connectivity cell data')
    parser.add_argument('--input_dir', '-i', required=True, action='store',
                        help='Directory that contains predicted data')
    parser.add_argument('--brain_seg_data_dir', '-b', required=True, action='store',
                        help='Directory that contains brain segmentation data')
    parser.add_argument('--output_dir', '-o', required=True, action='store',
                        help='Directory that will contain output')
    parser.add_argument('--structure_id', '-s', action='store', type=int, default=997,
                        help='Number of this instance')
    parser.add_argument('--verify_thumbnail', '-t', action='store_true', default=False,
                        help='Verify thumbnail/structure mask IOU')
    return parser


if __name__ == '__main__':
    parser = create_cell_build_argparser()
    parser.add_argument('--parallel_processors', '-p', action='store', type=int, required=True,
                        help='Number of parallel processors')
    args = parser.parse_args()

    print(vars(args))
    main(**vars(args))

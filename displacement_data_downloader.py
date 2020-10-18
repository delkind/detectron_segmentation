import argparse
import os

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from dir_watcher import DirWatcher


class SegmentationDataDownloader(DirWatcher):
    def __init__(self, output_dir, number, resolution):
        super().__init__(*[os.path.join(output_dir, d) for d in ['data/input', 'data/dl', 'data/ready']],
                         f'downloader-{number}')
        self.mcc = MouseConnectivityCache(manifest_file=f'{output_dir}/connectivity/mouse_connectivity_manifest.json',
                                          resolution=resolution)
        self.mcc.get_annotation_volume()
        self.mcc.get_reference_space()

    def process_item(self, item, directory):
        self.logger.info(f"Download displacement data for {item}")
        self.mcc.get_deformation_field(item, header_path=f'{directory}/dfmfld.mhd',
                                       voxel_path=f'{directory}/dfmfld.raw')
        self.mcc.get_affine_parameters(item, direction='trv', file_name=f'{directory}/aff_param.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download displacement data for Mouse Connectivity')
    parser.add_argument('--output_dir', '-o', required=True, action='store', help='Directory that will contain output')
    parser.add_argument('--number', '-n', action='store', type=int, required=True, help='Number of this instance')
    parser.add_argument('--resolution', '-r', required=True, type=int, action='store',
                        help='Reference space resolution')
    args = parser.parse_args()

    print(vars(args))
    loader = SegmentationDataDownloader(**vars(args))
    loader.run_until_empty()

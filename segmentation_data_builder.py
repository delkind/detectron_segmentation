import argparse
import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi

from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from dir_watcher import DirWatcher


class ExperimentSectionData(object):
    def __init__(self, mcc, experiment_id, output_dir, anno, meta, rsp, logger, zoom=8, remove_transform_data=True):
        self.remove_transform_data = remove_transform_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.mcc = mcc
        self.mapi = MouseConnectivityApi()
        self.anno, self.meta = anno, meta
        self.rsp = rsp
        self.zoom = 8 - zoom
        self.id = experiment_id
        assert zoom >= 0
        self.details = self.mapi.get_experiment_detail(self.id)
        image_resolution = self.details[0]['sub_images'][0]['resolution']
        self.two_d = 1.0 / image_resolution
        self.size = self.mcc.resolution * self.two_d / (2 ** self.zoom)
        self.dims = (self.details[0]['sub_images'][0]['height'] // (2 ** self.zoom),
                     self.details[0]['sub_images'][0]['width'] // (2 ** self.zoom))
        self.root_points = np.array(np.where(self.anno != 0)).T
        self.logger = logger
        self.logger.info(f"Initializing displacement transform data for {self.id}...")
        self.__init_transform__()
        self.logger.info(f"Performing displacement transformation for {self.id}...")
        self.__init_transformed_points__()

    def __init_transform__(self):
        temp = sitk.ReadImage(f'{self.output_dir}/dfmfld.mhd', sitk.sitkVectorFloat64)
        dfmfld_transform = sitk.DisplacementFieldTransform(temp)

        temp = self.mcc.get_affine_parameters(self.id, direction='trv', file_name=f'{self.output_dir}/aff_param.txt')
        aff_trans = sitk.AffineTransform(3)
        aff_trans.SetParameters(temp.flatten())

        self.transform = sitk.Transform(3, sitk.sitkComposite)
        self.transform.AddTransform(aff_trans)
        self.transform.AddTransform(dfmfld_transform)

    def __init_transformed_points__(self):
        self.transformed_points = self.__transform_points__(self.transform, self.root_points.astype(float) *
                                                            self.mcc.resolution)
        self.transformed_points[..., :2] *= self.two_d / (2 ** self.zoom)
        self.transformed_points[..., 2] /= 100
        self.next_points = self.transformed_points.copy()
        self.next_points[..., :2] += self.size
        self.transformed_points = np.round(self.transformed_points).astype(int)
        self.next_points = np.round(self.next_points).astype(int)

    @staticmethod
    def __transform_points__(composite_transform, points):
        return np.array(list(map(composite_transform.TransformPoint, points)))

    def create_section_data(self):
        first_section = np.min(self.transformed_points[..., -1])
        last_section = np.max(self.transformed_points[..., -1])
        result = np.zeros((*self.dims, last_section + 1), dtype=np.int32)

        self.logger.info(f"Transferring segmentation data for {self.id}...")
        transformed_indices = tuple(self.transformed_points.squeeze().T.tolist())
        next_indices = tuple(self.next_points.squeeze().T.tolist())
        original_indices = tuple(self.root_points.squeeze().T.tolist())
        result[transformed_indices[1], transformed_indices[0], transformed_indices[2]] = self.anno[original_indices]
        result[next_indices[1], next_indices[0], next_indices[2]] = self.anno[original_indices]

        structures = np.unique(result).tolist()
        structures = list(set(structures).difference({0}))
        sorted_indx = np.argsort(
            np.array(list(map(lambda x: len(x), self.rsp.structure_tree.ancestor_ids(structures)))))
        structures = np.array(structures)[sorted_indx].tolist()

        new_result = np.zeros_like(result)

        self.logger.info(f"Filling holes for {self.id}...")
        for i, struct in enumerate(structures):
            mask = result == struct
            mask = ndi.binary_closing(ndi.binary_fill_holes(mask).astype(np.int32)).astype(np.int32)
            new_result[mask != 0] = struct

        self.logger.info(f"Saving segmentation data for {self.id}...")
        np.savez_compressed(f"{self.output_dir}/{self.id}-sections", new_result)

    def cleanup(self):
        if self.remove_transform_data:
            os.remove(f'{self.output_dir}/dfmfld.mhd')
            os.remove(f'{self.output_dir}/dfmfld.raw')
            os.remove(f'{self.output_dir}/aff_param.txt')


class SegmentationDataBuilder(DirWatcher):
    def __init__(self, output_dir, resolution, retain_transform_data, zoom, number, count):
        super().__init__(*[os.path.join(output_dir, d) for d in ['data/ready', 'data/proc', 'data/result']],
                         f'segmentation-data-builder-{number}')
        self.count = count
        self.zoom = zoom
        self.retain_transform_data = retain_transform_data
        self.resolution = resolution
        self.output_dir = output_dir
        self.mcc = MouseConnectivityCache(manifest_file=f'{output_dir}/connectivity/mouse_connectivity_manifest.json',
                                          resolution=resolution)
        self.anno, self.meta = self.mcc.get_annotation_volume()
        self.rsp = self.mcc.get_reference_space()
        self.rsp.remove_unassigned()  # This removes ids that are not in this particular reference space

    def process_item(self, item, directory):
        item = int(item)
        experiment = ExperimentSectionData(self.mcc, item, directory, self.anno, self.meta, self.rsp,
                                           self.logger, zoom=self.zoom,
                                           remove_transform_data=not self.retain_transform_data)
        experiment.create_section_data()
        experiment.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build segmentation data for Mouse Connectivity')
    parser.add_argument('--output_dir', '-o', required=True, action='store', help='Directory that will contain output')
    parser.add_argument('--resolution', '-r', required=True, type=int, action='store',
                        help='Reference space resolution')
    parser.add_argument('--zoom', '-z', required=True, type=int, action='store', help='Image zoom')
    parser.add_argument('--retain_transform_data', '-t', action='store_true', default=False,
                        help='Retain the transform data')
    parser.add_argument('--number', '-n', action='store', type=int, required=True, help='Number of this instance')
    parser.add_argument('--count', '-c', action='store', type=int, required=True, help='Total count of experiments')
    args = parser.parse_args()

    print(vars(args))
    builder = SegmentationDataBuilder(**vars(args))
    builder.run_until_count(builder.count)

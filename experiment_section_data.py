import argparse
import os
import subprocess
from multiprocessing.pool import Pool

import SimpleITK as sitk
import cv2
import numpy as np
import scipy.ndimage as ndi
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


class ExperimentSectionData(object):
    def __init__(self, mcc, experiment_id, output_dir, anno, meta, rsp, zoom=8, remove_transform_data=True):
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
        print(f"Initializing displacement transform data for {self.id}...")
        self.__init_transform__(remove_transform_data)
        print(f"Performing displacement transformation for {self.id}...")
        self.__init_transformed_points__()

    def __init_transform__(self, remove_data):
        self.mcc.get_deformation_field(self.id, header_path=f'{self.output_dir}/dfmfld.mhd',
                                       voxel_path=f'{self.output_dir}/dfmfld.raw')
        temp = sitk.ReadImage(f'{self.output_dir}/dfmfld.mhd', sitk.sitkVectorFloat64)
        dfmfld_transform = sitk.DisplacementFieldTransform(temp)

        temp = self.mcc.get_affine_parameters(self.id, direction='trv', file_name=f'{self.output_dir}/aff_param.txt')
        aff_trans = sitk.AffineTransform(3)
        aff_trans.SetParameters(temp.flatten())

        self.transform = sitk.Transform(3, sitk.sitkComposite)
        self.transform.AddTransform(aff_trans)
        self.transform.AddTransform(dfmfld_transform)

        if remove_data:
            os.remove(f'{self.output_dir}/dfmfld.mhd')
            os.remove(f'{self.output_dir}/dfmfld.raw')
            os.remove(f'{self.output_dir}/aff_param.txt')

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

    @staticmethod
    def fill_holes(source):
        ret, thresh = cv2.threshold(source.astype(np.uint8) * 50, 5, 255, cv2.THRESH_BINARY)
        ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(source, dtype=np.uint8)
        cv2.fillPoly(mask, ctrs, color=255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.erode(mask, kernel, cv2.BORDER_CONSTANT)
        mask = cv2.dilate(mask, kernel, cv2.BORDER_CONSTANT)
        return mask

    def create_section_data(self):
        first_section = np.min(self.transformed_points[..., -1])
        last_section = np.max(self.transformed_points[..., -1])
        result = np.zeros((*self.dims, last_section + 1), dtype=np.int32)

        print(f"Transferring segmentation data for {self.id}...")
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

        print(f"Filling holes for {self.id}...")
        for i, struct in enumerate(structures):
            mask = result == struct
            mask = ndi.binary_closing(ndi.binary_fill_holes(mask).astype(np.int32)).astype(np.int32)
            new_result[mask != 0] = struct

        print(f"Saving segmentation data for {self.id}...")
        np.savez_compressed(f"{self.output_dir}/{self.id}-sections", new_result)


def main(output_dir, resolution, retain_transform_data, zoom, parallel_threads, number):
    mcc = MouseConnectivityCache(manifest_file=f'{output_dir}/{number}/connectivity/mouse_connectivity_manifest.json',
                                 resolution=resolution)
    experiments = sorted(mcc.get_experiments(dataframe=False), key=lambda e: e['id'])

    if number is None:
        processes = [subprocess.Popen(["python",
                                       "./experiment_section_data.py",
                                       f"-o {output_dir}", f"-r {resolution}",
                                       f"-z {zoom}"
                                       ] +
                                      (["-t"] if retain_transform_data else []) +
                                      [f"-p {parallel_threads}", f"-n {i}"])
                     for i in range(parallel_threads)]
        exit_codes = [p.wait() for p in processes]
    else:
        anno, meta = mcc.get_annotation_volume()
        rsp = mcc.get_reference_space()
        rsp.remove_unassigned()  # This removes ids that are not in this particular reference space
        for experiment in experiments[number::parallel_threads]:
            experiment_id = experiment['id']
            experiment = ExperimentSectionData(mcc, experiment_id, f'{output_dir}/{experiment_id}/', anno, meta, rsp,
                                               zoom=zoom, remove_transform_data=not retain_transform_data)
            experiment.create_section_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build segmentation data for Mouse Connectivity')
    parser.add_argument('--output_dir', '-o', required=True, action='store', help='Directory that will contain output')
    parser.add_argument('--resolution', '-r', default=25, type=int, action='store', help='Reference space resolution')
    parser.add_argument('--zoom', '-z', default=2, type=int, action='store', help='Image zoom')
    parser.add_argument('--retain_transform_data', '-t', action='store_true', default=False,
                        help='Retain the transform data')
    parser.add_argument('--parallel_threads', '-p', action='store', type=int, required=True,
                        help='Number of parallel threads')
    parser.add_argument('--number', '-n', action='store', type=int, default=None,
                        help='Number of this instance')
    args = parser.parse_args()

    print(vars(args))
    main(**vars(args))

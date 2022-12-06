import argparse
import copy
import json
import logging
import os

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader, \
    DatasetMapper
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from fvcore.common.file_io import PathManager
from PIL import Image


class MyDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        self.tfm_gens = self.tfm_gens + [T.RandomBrightness(0.5, 2),
                                         T.RandomContrast(0.5, 2),
                                         ]

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = image[:, :image.shape[0]]
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        DefaultTrainer.build_evaluator(cls, cfg, dataset_name)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=MyDatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MyDatasetMapper(cfg, True))


def get_balloon_dicts(img_dir, json_file, train=False):
    json_file = os.path.join(img_dir, json_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []

    images = list(imgs_anns['_via_img_metadata'].values())

    if train and all(map(lambda e: 'score' in e, images)):
        images = sorted(images, key=lambda e: e['score'])
        scores = np.array([i['score'] for i in images])
        images = images + images[:int((scores < 0.9).sum())]

    for idx, v in enumerate(images):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = height

        annos = v["regions"]
        objs = []
        for anno in annos:
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def main(image_dir, project, crop_size, batch_size, iterations, validation_split, backbone,
         output_dir, learning_rate, device, tb, tb_port, resume=None):
    if tb:
        print(f'Tensorboard URL: {launch_tb(output_dir, tb_port)}')

    DatasetCatalog.register("train", lambda: get_balloon_dicts(image_dir, project, True))
    MetadataCatalog.get("train").set(thing_classes=["balloon"])
    DatasetCatalog.register("val", lambda: get_balloon_dicts(image_dir, project, False))
    MetadataCatalog.get("val").set(thing_classes=["balloon"])
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train", )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    if resume:
        cfg.MODEL.WEIGHTS = resume
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            f"COCO-InstanceSegmentation/mask_rcnn_{backbone}.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate  # pick a good LR
    cfg.SOLVER.MAX_ITER = iterations  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.OUTPUT_DIR = output_dir
    cfg.INPUT.CROP.SIZE = crop_size
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def launch_tb(output_dir, port):
    from tensorboard import program
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--bind_all', '--logdir', output_dir, '--port', port])
    return tb.launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detectron Mask R-CNN for cells counting and segmentation - training')

    parser.add_argument('--image_dir', action='store', required=True)
    parser.add_argument('--project', action='store', required=True)
    parser.add_argument('--crop_size', action='store', default=312, type=int)
    parser.add_argument('--batch_size', default=2, type=int, action='store', help='Some help')
    parser.add_argument('--iterations', default=1000000, type=int, action='store', help='Some help')
    parser.add_argument('--validation_split', default=10, type = int, action='store', help='Some help')
    parser.add_argument('--backbone', default='R_50_FPN_3x', action='store', help='Some help')
    parser.add_argument('--output_dir', required=True, action='store', help='Some help')
    parser.add_argument('--resume', required=False, action='store', help='Some help')
    parser.add_argument('--learning_rate', default=0.00025, type=float, action='store', help='Some help')
    parser.add_argument('--device', default='cuda', action='store', help='Model execution device')
    parser.add_argument('--tb', default=False, action='store_true', help='Start tensorboard')
    parser.add_argument('--tb_port', default='6006', action='store', help='Tensorboard port')
    args = parser.parse_args()
    print(vars(args))

    main(**vars(args))

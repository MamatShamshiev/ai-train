import os

import albumentations as albu
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
import wandb
from detectron2.config.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import default_argument_parser, launch
from detectron2.engine.defaults import DefaultTrainer, default_setup
from detectron2.evaluation import COCOEvaluator

import dt2.models  # noqa
from defs import (
    TRAIN_DATA_PATH,
    TRAIN_IMAGES_PATH,
    TRAIN_MASKS_PATH,
    VAL_DATA_PATH,
    VAL_IMAGES_PATH,
    VAL_MASKS_PATH,
)
from dt2.hooks import BestCheckpointerHook, NumberOfParamsHook, PredsVisHook
from dt2.register import register_my_dataset
from dt2.transforms.augmentations import albu_to_dt2_aug


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_mapper(cls, cfg):
        """
        albu_tfms = [
            (albu.CLAHE(always_apply=True), 0.2),
            (albu.HueSaturationValue(always_apply=True), 0.2),
            (albu.RandomGamma(always_apply=True), 0.2),
            (albu.GaussNoise(always_apply=True), 0.2),
            (albu.Blur(always_apply=True), 0.2),
            (albu.MotionBlur(always_apply=True), 0.2),
            (albu.ISONoise(always_apply=True), 0.2),
            (albu.Sharpen(always_apply=True), 0.2),
            (albu.ImageCompression(quality_lower=60, always_apply=True), 0.2),
            (albu.Downscale(scale_min=0.7, scale_max=0.95, always_apply=True), 0.2),
        ]
        """
        albu_tfms = []
        albu_augs = [albu_to_dt2_aug(tfm, prob=prob) for tfm, prob in albu_tfms]

        dt2_aigs = [
            T.RandomApply(T.RandomCrop("relative_range", [0.8, 0.8]), prob=0.5),
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style="choice",
            ),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
        ]
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=dt2_aigs + albu_augs,
        )
        return mapper

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = cls.build_train_mapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointerHook(self.cfg.TEST.EVAL_PERIOD))

        dataset_dicts_train = DatasetCatalog.get(self.cfg.DATASETS.TRAIN[0])
        train_mapper = self.build_train_mapper(self.cfg)
        batches_train = [train_mapper(d) for d in dataset_dicts_train[:1]]
        dataset_dicts_test = DatasetCatalog.get(self.cfg.DATASETS.TEST[0])
        test_mapper = DatasetMapper(self.cfg, is_train=True)
        batches_test = [test_mapper(d) for d in dataset_dicts_test[:1]]
        hooks.insert(
            -1,
            PredsVisHook(
                batches_train,
                batches_test,
                MetadataCatalog.get(self.cfg.DATASETS.TEST[0]),
                self.cfg.TEST.EVAL_PERIOD,
                self.cfg.INPUT.FORMAT,
            ),
        )
        hooks.insert(-1, NumberOfParamsHook())
        return hooks


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_my_dataset(
        "train-dataset",
        {},
        TRAIN_IMAGES_PATH,
        TRAIN_IMAGES_PATH,
        TRAIN_MASKS_PATH,
        TRAIN_DATA_PATH / "coco.json",
    )

    register_my_dataset(
        "val-dataset",
        {},
        VAL_IMAGES_PATH,
        VAL_IMAGES_PATH,
        VAL_MASKS_PATH,
        VAL_DATA_PATH / "coco.json",
    )

    cfg = setup(args)
    if comm.is_main_process():
        project_name = str(cfg.OUTPUT_DIR)
        project_name = project_name[:-1] if project_name.endswith("/") else project_name
        run_name = project_name.split("/")[-1]
        wandb.init(project="ai-train", name=run_name, sync_tensorboard=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

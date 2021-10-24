import os

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
import wandb
from detectron2.config.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.build import build_detection_test_loader
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
from dt2.hooks import (
    BestCheckpointerHook,
    LossEvalHook,
    NumberOfParamsHook,
    PredsVisHook,
)
from dt2.register import register_my_dataset


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.RandomCrop("relative_range", [0.7, 0.7]),
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
                T.RandomSaturation(0.8, 1.2),
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointerHook(self.cfg.TEST.EVAL_PERIOD))
        train_loader = self.build_train_loader(self.cfg)
        batch_train = next(iter(train_loader))
        test_loader = build_detection_test_loader(
            dataset=DatasetCatalog.get(self.cfg.DATASETS.TEST[0]),
            mapper=DatasetMapper(self.cfg, is_train=True),
        )
        batch_test = next(iter(test_loader))
        hooks.insert(
            -1,
            PredsVisHook(
                batch_train,
                batch_test,
                MetadataCatalog.get(self.cfg.DATASETS.TEST[0]),
                self.cfg.TEST.EVAL_PERIOD,
                self.cfg.INPUT.FORMAT,
            ),
        )
        # if (
        #     self.cfg.SOLVER.REFERENCE_WORLD_SIZE == 1
        # ):  # this hook somehow works only on a single-gpu
        #     hooks.insert(-1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD, test_loader))
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

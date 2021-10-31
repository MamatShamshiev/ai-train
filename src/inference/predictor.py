from pathlib import Path

import dt2.modeling  # noqa
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling.meta_arch.build import build_model


def get_exp_cfg(
    exp_dir: Path,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_image: int = 200,
):
    cfg = get_cfg()
    cfg.merge_from_file(exp_dir / "config.yaml")
    cfg.MODEL.WEIGHTS = str(exp_dir / "model_best.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = nms_thresh
    cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image
    return cfg


def get_predictor(
    exp_dir: Path,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_image: int = 200,
) -> DefaultPredictor:
    cfg = get_exp_cfg(exp_dir, score_thresh, nms_thresh, detections_per_image)
    predictor = DefaultPredictor(cfg)
    return predictor


def get_model_dict_for_inference(
    exp_dir: Path,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_image: int = 200,
):
    cfg = get_exp_cfg(exp_dir, score_thresh, nms_thresh, detections_per_image)
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return {
        "model": model,
        "min_size": cfg.INPUT.MIN_SIZE_TEST,
        "max_size": cfg.INPUT.MAX_SIZE_TEST,
    }

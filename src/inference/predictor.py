from pathlib import Path

from detectron2.config.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor


def get_predictor(
    exp_dir: Path,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_image: int = 200,
) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(exp_dir / "config.yaml")
    cfg.MODEL.WEIGHTS = str(exp_dir / "model_best.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
    cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image
    predictor = DefaultPredictor(cfg)
    return predictor

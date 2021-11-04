from pathlib import Path

import dt2.modeling  # noqa
import torch
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling.meta_arch.build import build_model
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from dt2.config import add_retina_config
from dt2.modeling.test_time_augmentation import MyPanopticFPNWithTTA


def get_exp_cfg(
    exp_dir: Path,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_image: int = 200,
):
    cfg = get_cfg()
    add_retina_config(cfg)
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
    tta: bool = False,
) -> DefaultPredictor:
    cfg = get_exp_cfg(exp_dir, score_thresh, nms_thresh, detections_per_image)
    if tta is True:
        predictor = TTAPredictor(cfg)
    else:
        predictor = DefaultPredictor(cfg)
    return predictor


def get_model_dict_for_inference(
    exp_dir: Path,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_image: int = 300,
    tta: bool = False,
):
    cfg = get_exp_cfg(exp_dir, score_thresh, nms_thresh, detections_per_image)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    if tta is True:
        if cfg.MODEL.META_ARCHITECTURE == "MyPanopticFPN":
            model = MyPanopticFPNWithTTA(cfg, model)
        elif cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
            model = GeneralizedRCNNWithTTA(cfg, model)
        else:
            print(f"Unable perform a TTA for meta arch {cfg.MODEL.META_ARCHITECTURE}")
    model.eval()
    return {
        "model": model,
        "min_size": cfg.INPUT.MIN_SIZE_TEST,
        "max_size": cfg.INPUT.MAX_SIZE_TEST,
    }


class TTAPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.MODEL.META_ARCHITECTURE == "MyPanopticFPN":
            self.model = MyPanopticFPNWithTTA(cfg, self.model)
        elif cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
            self.model = GeneralizedRCNNWithTTA(cfg, self.model)
        else:
            raise ValueError(cfg.MODEL.META_ARCHITECTURE)
        self.model.eval()

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

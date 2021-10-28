from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from tqdm import tqdm

from inference.ensemble import ENSEMBLE_METHOD, BoxEnsembler
from inference.predictor import get_predictor
from inference.utils import prepare_detection_submit, prepare_segmentation_submit


def get_predictions_for_submit(
    path_to_images: Path,
    exp_dirs: List[Path],
    score_threshes: List[float],
    nms_threshes: List[float],
    ensemble_method: ENSEMBLE_METHOD = "wbf",
    **ensemble_kwargs,
) -> Tuple[List[Dict], List[torch.Tensor]]:
    """Run inference and prepare predictions for submission.

    Args:
        path_to_images (Path): path to images folder
        exp_dirs (List[Path]): list of experiment directories.
            Each directory must contain 'config.yaml' and 'model_best.pth' files.
        score_threshes (List[float]): score thresholds for each experiment.
        nms_threshes (List[float]): nms thresholds for each experiment.
        ensemble_method (ENSEMBLE_METHOD, optional): ensemble method.
            Must be one of "nms", "soft_nms", "nmw", "wbf". Defaults to "wbf".
        ensemble_kwargs: keyword arguments for ensembling method.


    Returns:
        Tuple[List[Dict], List[torch.Tensor]]: detection predictions and segmentation predictions
    """
    assert len(exp_dirs) == len(score_threshes) == len(nms_threshes)
    image_paths = sorted(list(Path(path_to_images).glob("*.png")))
    detection_predictions = []
    segm_predictions = []

    box_ensembler = BoxEnsembler()
    predictor_by_exp = {
        exp_dir: get_predictor(exp_dir, score_thresh, nms_thresh)
        for exp_dir, score_thresh, nms_thresh in zip(
            exp_dirs, score_threshes, nms_threshes
        )
    }

    for path_to_img in tqdm(image_paths):
        prediction = {"image_id": int(path_to_img.name.split(".")[1])}
        im = cv2.imread(str(path_to_img))
        sem_seg = torch.zeros(
            (4, *im.shape[:2]), device=torch.device("cuda:0"), requires_grad=False
        )  # C x H x W
        instances_list = []
        for exp_dir in exp_dirs:
            predictor = predictor_by_exp[exp_dir]

            outputs = predictor(im)
            instances = outputs["instances"].to("cpu")
            instances_list.append(instances)
            if "sem_seg" in outputs:
                sem_seg += outputs["sem_seg"]

        if len(exp_dirs) == 1:
            instances = instances_list[0]
        else:
            instances = box_ensembler(
                instances_list, ensemble_method, **ensemble_kwargs
            )
        prediction["instances"] = instances_to_coco_json(
            instances, prediction["image_id"]
        )
        detection_predictions.append(prediction)
        sem_seg = torch.argmax(sem_seg, axis=0).cpu().numpy().astype(np.uint8)
        segm_predictions.append(sem_seg)
        assert sem_seg.shape == im.shape[:2]
    return detection_predictions, segm_predictions


def prepare_submit(
    path_to_images: Path,
    path_to_save: Path,
    exp_dirs: List[Path],
    score_threshes: List[float],
    nms_threshes: List[float],
    ensemble_method: ENSEMBLE_METHOD = "wbf",
    **ensemble_kwargs,
) -> None:
    image_paths = sorted(list(Path(path_to_images).glob("*.png")))
    detection_predictions, segm_predictions = get_predictions_for_submit(
        path_to_images=path_to_images,
        exp_dirs=exp_dirs,
        score_threshes=score_threshes,
        nms_threshes=nms_threshes,
        ensemble_method=ensemble_method,
        **ensemble_kwargs,
    )

    prepare_detection_submit(
        detection_predictions, str(path_to_save / "detection_predictions.json")
    )

    prepare_segmentation_submit(
        image_paths, segm_predictions, path_to_save / "segmentation_predictions.json"
    )

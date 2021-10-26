from pathlib import Path

import cv2
import torch
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from tqdm import tqdm

from predictor import get_predictor
from utils import prepare_detection_submit, prepare_segmentation_submit


def prepare_submit(
    path_to_images: Path,
    path_to_save: Path,
    exp_dir: Path,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
) -> None:
    predictor = get_predictor(exp_dir, score_thresh, nms_thresh)
    image_paths = sorted(list(Path(path_to_images).glob("*.png")))

    detection_predictions = []
    segm_predictions = []
    for path_to_img in tqdm(image_paths):
        prediction = {"image_id": int(path_to_img.name.split(".")[1])}
        im = cv2.imread(str(path_to_img))
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        prediction["instances"] = instances_to_coco_json(
            instances, prediction["image_id"]
        )
        detection_predictions.append(prediction)

        with torch.no_grad():
            sem_seg = torch.argmax(outputs["sem_seg"], axis=0).cpu().numpy()
        segm_predictions.append(sem_seg)
        assert sem_seg.shape == im.shape[:2]

    prepare_detection_submit(
        detection_predictions, str(path_to_save / "detection_predictions.json")
    )

    prepare_segmentation_submit(
        image_paths, segm_predictions, path_to_save / "segmentation_predictions.json"
    )

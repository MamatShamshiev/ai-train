import itertools
from pathlib import Path
from typing import Dict, List, Tuple

from baseline.evaluation.evaluation import calculation_map_050
from defs import VAL_DATA_PATH

from inference.ensemble import ENSEMBLE_METHOD
from inference.submit import get_predictions_for_submit
from inference.utils import prepare_detection_submit


def run_grid_search(
    path_to_images: Path,
    path_to_save: Path,
    exp_dirs: List[Path],
    score_threshes_list: List[List[float]],
    nms_threshes_list: List[List[float]],
    ensemble_methods_list: List[ENSEMBLE_METHOD],
    ensemble_kwargs_list: List[Dict],
) -> None:
    for (
        score_threshes,
        nms_threshes,
        ensemble_method,
        ensemble_kwargs,
    ) in itertools.product(
        *[
            score_threshes_list,
            nms_threshes_list,
            ensemble_methods_list,
            ensemble_kwargs_list,
        ]
    ):
        detection_predictions, _ = get_predictions_for_submit(
            path_to_images,
            exp_dirs,
            score_threshes=score_threshes,
            nms_threshes=nms_threshes,
            ensemble_method=ensemble_method,
            **ensemble_kwargs
        )
        prepare_detection_submit(
            detection_predictions, str(path_to_save / "detection_predictions.json")
        )
        metric = calculation_map_050(
            str(VAL_DATA_PATH / "coco.json"),
            str(path_to_save / "detection_predictions.json"),
        )
        print("______________________________")
        print(score_threshes, nms_threshes, ensemble_method, ensemble_kwargs)
        print(metric)
        print("______________________________")

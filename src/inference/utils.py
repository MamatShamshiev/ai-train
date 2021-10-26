import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pycocotools.mask as pctmask
from tqdm import tqdm

ALTERNATIVE_RAIL_POLYGON = 1
MAIN_RAIL_POLYGON = 2
TRAIN = 3


def prepare_detection_submit(detection_predictions: List[Dict], path_to_save: str):
    coco_results = list(
        itertools.chain(*[x["instances"] for x in detection_predictions])
    )
    for result in coco_results:
        result["category_id"] += 1
    with open(path_to_save, "w") as f:
        f.write(json.dumps(coco_results))


def prepare_segmentation_submit(
    image_paths: List[Path], predicted_masks: List[np.ndarray], path_to_save: str
) -> None:
    """Function to prepare json with encoded masks, ready for evaluation.
       !Note!: Each N_i mask of predicted_masks have to correspond to N_i image name in image_filenames list!

    Args:
        image_paths (List[Path]): N paths to images used for an inference.
        predicted_masks (np.ndarray): numpy array of (N, height, width, 3 (corresponds to number of classes)). Proper width and height will be taken from original images.
        path_to_save (str): where to store json with predictions
    """

    # Pay attention, your model may output in another order, but class index have to match with this structure
    submit = {
        "images": [],
        "categories": [
            {"supercategory": "railway_object", "id": 0, "name": "MainRailPolygon"},
            {
                "supercategory": "railway_object",
                "id": 1,
                "name": "AlternativeRailPolygon",
            },
            {"supercategory": "railway_object", "id": 2, "name": "Train"},
        ],
    }
    masks_list = []
    for mask in predicted_masks:
        image_annots = []
        for c_id, c in enumerate((ALTERNATIVE_RAIL_POLYGON, MAIN_RAIL_POLYGON, TRAIN)):
            _single_mask = mask == c
            encoded = pctmask.encode(np.asfortranarray(_single_mask).astype(np.uint8))
            encoded["class_id"] = c_id
            image_annots.append(encoded)
        masks_list.append(image_annots)

    image_filenames = [path.name for path in image_paths]
    path_to_save = Path(path_to_save)

    for filename, mask in tqdm(
        zip(image_filenames, masks_list),
        total=len(image_filenames),
        desc="segmentation submit preparation: ",
    ):
        assert len(mask) == 3, "Mask have to contain predicts for all 3 classes."

        image_annots = []
        for class_id in (0, 1, 2):
            encoded = mask[class_id]
            image_annots.append(
                {
                    "counts": encoded["counts"].decode("ascii"),
                    "size": encoded["size"],
                    "class_id": encoded["class_id"],
                }
            )

        submit["images"].append({"file_name": filename, "annotations": image_annots})
    with open(path_to_save, "w") as json_file:
        json.dump(submit, json_file)

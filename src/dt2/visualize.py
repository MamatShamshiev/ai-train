import copy
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from data.prepare_masks import process_for_dt2_visualization, read_mask
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer

THING_COLORS = [
    (100, 170, 150),
    (139, 106, 38),
    (22, 213, 59),
    (10, 64, 86),
    (116, 146, 4),
    (4, 146, 170),
    (44, 254, 36),
    (104, 253, 109),
    (225, 9, 209),
    (71, 120, 135),
    (17, 246, 227),
]

STUFF_COLORS = [(17, 209, 234), (80, 60, 169), (151, 154, 224)]


def visualize_dataset_dict(
    d: Dict[str, Any],
    metadata: Metadata,
    scale: float = 1.0,
    plot: bool = True,
    figsize=(30, 20),
) -> np.ndarray:
    d = copy.deepcopy(d)
    for item in d["annotations"]:
        item.pop("segmentation")

    sem_seg_file_name = d.pop("sem_seg_file_name")
    img = cv2.imread(d["file_name"])[:, :, ::-1]
    if plot is True:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()

    visualizer = Visualizer(img, metadata=metadata, scale=scale)
    out = visualizer.draw_dataset_dict(d)
    mask = read_mask(sem_seg_file_name)
    mask = process_for_dt2_visualization(mask)
    out = visualizer.draw_sem_seg(mask)
    out = out.get_image()
    if plot is True:
        plt.figure(figsize=figsize)
        plt.imshow(out)
        plt.show()
    return out


def visualize_batch_item(
    batch_item: Dict[str, Any], metadata: Metadata, plot: bool = True, figsize=(30, 20)
):
    # image must be RGB
    image = batch_item["image"].permute(1, 2, 0).numpy()
    if not batch_item["instances"].has("pred_boxes"):
        batch_item["instances"].pred_boxes = batch_item["instances"].gt_boxes
    if not batch_item["instances"].has("pred_classes"):
        batch_item["instances"].pred_classes = batch_item["instances"].gt_classes
    visualizer = Visualizer(image, metadata=metadata, scale=1)
    out = visualizer.draw_instance_predictions(batch_item["instances"].to("cpu"))

    if "sem_seg" in batch_item:
        mask = batch_item["sem_seg"].clone()
        mask = process_for_dt2_visualization(mask)
        out = visualizer.draw_sem_seg(mask.to("cpu"))
    out = out.get_image()
    if plot is True:
        plt.figure(figsize=figsize)
        plt.imshow(out)
        plt.show()
    return out

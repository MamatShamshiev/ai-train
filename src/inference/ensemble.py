from typing import List, Tuple

import ensemble_boxes
import torch
from detectron2.structures import Boxes, Instances
from scipy.stats import rankdata
from typing_extensions import Literal

ENSEMBLE_METHOD = Literal["nms", "soft_nms", "nmw", "wbf"]


def convert_instances_to_wbf_format(instances: Instances) -> Tuple[List, List, List]:
    image_h, image_w = instances.image_size
    boxes = instances.pred_boxes.clone()
    boxes.scale(scale_x=1 / image_w, scale_y=1 / image_h)
    boxes = boxes.tensor.tolist()
    scores = instances.scores.tolist()
    labels = instances.pred_classes.tolist()
    return boxes, scores, labels


def convert_wbf_format_to_instances(
    boxes: List, scores: List, labels: List, image_h: int, image_w: int
):
    pred_boxes = Boxes(boxes)
    pred_boxes.scale(scale_x=image_w, scale_y=image_h)
    scores = torch.as_tensor(scores)
    pred_classes = torch.as_tensor(labels)
    instances = Instances(
        image_size=(image_h, image_w),
        pred_boxes=pred_boxes,
        scores=scores,
        pred_classes=pred_classes,
    )
    return instances


class BoxEnsembler:
    def __call__(
        self,
        instances_list: List[Instances],
        method_name: ENSEMBLE_METHOD,
        normalize_scores: bool = False,
        **kwargs,
    ) -> Instances:
        boxes_list, scores_list, labels_list = [], [], []
        for instances in instances_list:
            boxes, scores, labels = convert_instances_to_wbf_format(instances)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        boxes, scores, labels = self.ensemble(
            boxes_list,
            scores_list,
            labels_list,
            method_name,
            normalize_scores,
            **kwargs,
        )
        instances = convert_wbf_format_to_instances(
            boxes,
            scores,
            labels,
            instances_list[0].image_size[0],
            instances_list[0].image_size[1],
        )
        return instances

    def ensemble(
        self,
        boxes_list: List[List[float]],
        scores_list: List[float],
        labels_list: List[int],
        method_name: ENSEMBLE_METHOD,
        normalize_scores: bool = False,
        **kwargs,
    ):
        method = self._get_method_by_name(method_name)
        if normalize_scores is True:
            scores_list = [
                0.5 * (rankdata(scores) / len(scores)) + 0.5 for scores in scores_list
            ]
        boxes, scores, labels = method(boxes_list, scores_list, labels_list, **kwargs)
        return boxes, scores, labels

    @classmethod
    def _get_method_by_name(cls, method_name: ENSEMBLE_METHOD):
        if method_name == "nms":
            method = ensemble_boxes.nms
        elif method_name == "soft_nms":
            method = ensemble_boxes.soft_nms
        elif method_name == "nmw":
            method = ensemble_boxes.non_maximum_weighted
        elif method_name == "wbf":
            method = ensemble_boxes.weighted_boxes_fusion
        else:
            raise ValueError(method_name)
        return method

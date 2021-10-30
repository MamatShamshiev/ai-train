from typing import Dict, List

import torch
import torch.nn as nn
from detectron2.config.config import configurable
from detectron2.layers import cat
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.meta_arch import PanopticFPN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.structures.image_list import ImageList
from detectron2.utils.events import get_event_storage
from fvcore.nn import sigmoid_focal_loss_jit
from losses import sigmoid_soft_focal_loss_jit
from torch.nn import functional as F


@META_ARCH_REGISTRY.register()
class MyPanopticFPN(PanopticFPN):
    def inference(
        self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, see docs in :meth:`forward`.
            Otherwise, returns a (list[Instances], list[Tensor]) that contains
            the raw detector outputs, and raw semantic segmentation outputs.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        sem_seg_results, _ = self.sem_seg_head(features, None)
        proposals, _ = self.proposal_generator(images, features, None)
        detector_results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(
                    sem_seg_result, image_size, height, width
                )
                detector_r = detector_postprocess(detector_result, height, width)
                processed_results.append(
                    {"sem_seg": sem_seg_r, "instances": detector_r}
                )
            return processed_results
        else:
            return detector_results, sem_seg_results


@META_ARCH_REGISTRY.register()
class RetinaNetWSemseg(RetinaNet):
    @configurable
    def __init__(
        self,
        *,
        sem_seg_head: nn.Module,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sem_seg_head = sem_seg_head
        self.label_smoothing = label_smoothing

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["sem_seg_head"] = build_sem_seg_head(cfg, ret["backbone"].output_shape())
        ret["label_smoothing"] = cfg.MODEL.RETINANET.LABEL_SMOOTHING
        return ret

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features_list = [features[f] for f in self.head_in_features]
        predictions = self.head(features_list)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert (
                "instances" in batched_inputs[0]
            ), "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            assert "sem_seg" in batched_inputs[0]
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg,
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value,
            ).tensor
            _, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)

            losses = self.forward_training(
                images, features_list, predictions, gt_instances
            )
            losses.update(sem_seg_losses)
            return losses
        else:
            detector_results = self.forward_inference(
                images, features_list, predictions
            )
            sem_seg_results, _ = self.sem_seg_head(features, None)
            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(
                    sem_seg_result, image_size, height, width
                )
                detector_r = detector_postprocess(detector_result, height, width)
                processed_results.append(
                    {"sem_seg": sem_seg_r, "instances": detector_r}
                )
            return processed_results

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        with torch.no_grad():
            # classification and regression loss
            # no loss for the last (background) class
            if self.label_smoothing == 0 or self.label_smoothing is None:
                gt_labels_target = F.one_hot(
                    gt_labels[valid_mask], num_classes=self.num_classes + 1
                )[:, :-1]
                loss_cls_func = sigmoid_focal_loss_jit
            else:
                smoothing = self.label_smoothing / self.num_classes
                gt_labels_target = torch.full(
                    (valid_mask.sum(), self.num_classes + 1),
                    smoothing,
                    device=gt_labels.device,
                )
                gt_labels_target.scatter_(
                    1, gt_labels[valid_mask].unsqueeze(1), 1 - smoothing
                )
                gt_labels_target = gt_labels_target[:, :-1]
                loss_cls_func = sigmoid_soft_focal_loss_jit

        loss_cls = loss_cls_func(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        return {
            "loss_cls": loss_cls / normalizer,
            "loss_box_reg": loss_box_reg / normalizer,
        }

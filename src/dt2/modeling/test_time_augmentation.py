from typing import List

import torch
from detectron2.config.config import configurable
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from dt2.modeling.panoptic_fpn import MyPanopticFPN
from dt2.transforms.augmentations import (
    FLIP_LABEL_MAPPING,
    HFlipTorch,
    ResizeShortestEdgeTorch,
)
from dt2.transforms.transforms import HFlipTransformTorch
from fvcore.transforms.transform import TransformList


class MyPanopticFPNWithTTA(GeneralizedRCNNWithTTA):
    def __init__(
        self,
        cfg,
        model,
        tta_mapper=None,
        batch_size=1,
        scales=(0.8, 1, 1.2),
        flip=False,
        flip_label_mapping=FLIP_LABEL_MAPPING,
    ):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        GeneralizedRCNNWithTTA.__base__.__init__(self)  # init torch.nn.Module
        assert isinstance(model, MyPanopticFPN), type(model)
        self.cfg = cfg.clone()
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"
        self.model = model
        if tta_mapper is None:
            tta_mapper = DatasetMapperTTATorch(
                min_sizes=[int(scale * cfg.INPUT.MIN_SIZE_TEST) for scale in scales],
                max_size=int(max(scales) * cfg.INPUT.MAX_SIZE_TEST),
                flip=flip,
            )
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size
        self.flip_label_mapping = flip_label_mapping

    def _batch_inference(self, batched_inputs):
        outputs = []
        inputs = []
        for idx, input in enumerate(batched_inputs):
            inputs.append(input)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.append(
                    self.model.inference(
                        inputs,
                        do_postprocess=False,
                    )
                )
                inputs = []
        return outputs

    def _get_augmented_outputs(self, augmented_inputs, tfms):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
        final_semseg = None

        for (detector_results, sem_seg_results, img_sizes), tfm in zip(outputs, tfms):
            # Need to inverse the transforms on boxes and semseg, to obtain results on original image
            detector_result = detector_results[0]
            img_size = img_sizes[0]
            sem_seg_result = sem_seg_results[0, :, : img_size[0], : img_size[1]]
            pred_boxes = detector_result.pred_boxes.tensor
            original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
            all_boxes.append(
                torch.from_numpy(original_pred_boxes).to(pred_boxes.device)
            )

            sem_seg_result = tfm.inverse().apply_image(sem_seg_result)
            if final_semseg is None:
                final_semseg = sem_seg_result
            else:
                final_semseg += sem_seg_result

            all_scores.extend(detector_result.scores)
            pred_classes = detector_result.pred_classes
            if (
                any(isinstance(t, HFlipTransformTorch) for t in tfm.transforms)
                and self.flip_label_mapping is not None
            ):
                self.flip_label_mapping = self.flip_label_mapping.to(
                    pred_classes.device
                )
                pred_classes = self.flip_label_mapping[pred_classes]
            all_classes.extend(pred_classes)

        all_boxes = torch.cat(all_boxes, dim=0)
        final_semseg /= len(tfms)
        return all_boxes, all_scores, all_classes, final_semseg

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        all_boxes, all_scores, all_classes, final_semseg = self._get_augmented_outputs(
            augmented_inputs, tfms
        )
        # merge all detected boxes to obtain final predictions for boxes
        merged_instances = self._merge_detections(
            all_boxes, all_scores, all_classes, orig_shape
        )

        sem_seg_r = sem_seg_postprocess(
            final_semseg, final_semseg.shape[-2:], *orig_shape
        )
        detector_r = detector_postprocess(merged_instances, *orig_shape)
        return {"sem_seg": sem_seg_r, "instances": detector_r}


class DatasetMapperTTATorch:
    @configurable
    def __init__(self, min_sizes: List[int], max_size: int, flip: bool):
        """
        Args:
            min_sizes: list of short-edge size to resize the image to
            max_size: maximum height or width of resized images
            flip: whether to apply flipping augmentation
        """
        self.min_sizes = min_sizes
        self.max_size = max_size
        self.flip = flip

    @classmethod
    def from_config(cls, cfg):
        return {
            "min_sizes": cfg.TEST.AUG.MIN_SIZES,
            "max_size": cfg.TEST.AUG.MAX_SIZE,
            "flip": cfg.TEST.AUG.FLIP,
        }

    def __call__(self, dataset_dict):
        torch_image = dataset_dict["image"]
        shape = torch_image.shape[-2:]
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        assert shape == orig_shape, (shape, orig_shape)

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdgeTorch(min_size, self.max_size)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = HFlipTorch()
                aug_candidates.append([resize, flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug_list in aug_candidates:
            tmfs = []
            new_image = torch_image
            for aug in aug_list:
                tfm = aug.get_transform(new_image)
                new_image = tfm.apply_image(new_image)
                tmfs.append(tfm)
            tfms = TransformList(tmfs)
            dic = {"height": dataset_dict["height"], "width": dataset_dict["width"]}
            dic["image"] = new_image
            dic["transforms"] = tfms
            ret.append(dic)
        return ret

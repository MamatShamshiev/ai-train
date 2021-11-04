import torch
from detectron2.data import transforms as T

from . import transforms as custom_T


class AlbuImageOnlyAugmentation(T.Augmentation):
    """
    Generic augmentation based on albumentations image only augs
    Args:
        prob (float) - probability to apply
        albu_transform - an instance of albumentation aug class
    Example:
        blur_aug = AlbuImageOnlyAugmentation(albu.Blur(), prob=0.2)
    """

    def __init__(self, albu_transform, prob=0.5):
        super().__init__()
        self.transform = albu_transform
        self.prob = prob

    def get_transform(self, image):
        do = self._rand_range() < self.prob
        if do:
            params = self.transform.get_params()
            if self.transform.targets_as_params:
                assert self.transform.targets_as_params == [
                    "image"
                ], f"{self.transform} is not an image-only transform since it has targets {self.transform.targets_as_params}!"
                targets_as_params = {"image": image}
                params_dependent_on_targets = (
                    self.transform.get_params_dependent_on_targets(targets_as_params)
                )
                params.update(params_dependent_on_targets)
            return custom_T.AlbuImageOnlyTransform(self.transform, params)
        else:
            return T.NoOpTransform()


def albu_to_dt2_aug(albu_transform, prob):
    return AlbuImageOnlyAugmentation(albu_transform=albu_transform, prob=prob)


class ResizeShortestEdgeTorch(T.Augmentation):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def _get_new_size(self, h, w):
        scale = self.min_size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.min_size, scale * w
        else:
            newh, neww = scale * h, self.min_size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return newh, neww

    def get_transform(self, img: torch.Tensor):
        assert len(img.shape) == 3  # C x H x W
        h, w = img.shape[-2:]
        new_h, new_w = self._get_new_size(h, w)
        return custom_T.ResizeTransformTorch(h, w, new_h, new_w)


class HFlipTorch(T.Augmentation):
    def get_transform(self, image: torch.Tensor):
        assert len(image.shape) == 3  # C x H x W
        w = image.shape[2]
        return custom_T.HFlipTransformTorch(w)


FLIP_LABEL_MAPPING = torch.LongTensor([0, 1, 2, 4, 3, 5, 7, 6, 8, 9, 10])
"""
Corresponds to the following classes used in dataset
'Car',
'Human',
'Wagon',
'FacingSwitchL',
'FacingSwitchR',
'FacingSwitchNV',
'TrailingSwitchL',
'TrailingSwitchR',
'TrailingSwitchNV',
'SignalE',
'SignalF'
"""

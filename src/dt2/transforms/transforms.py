import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data import transforms as T


class AlbuImageOnlyTransform(T.Transform):
    """
    Internal class. If you want to use albumentations image-only aug,
    you should use .augmentations.AlbuImageOnlyAugmentation class
    """

    def __init__(self, albu_transform, params):
        self.transform = albu_transform
        self.params = params

    def apply_image(self, img):
        return self.transform.apply(img, **self.params)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation


class ResizeShortestEdgeGPU:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

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

    def __call__(self, img: np.ndarray):
        h, w = img.shape[:2]
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        new_h, new_w = self._get_new_size(h, w)
        if any(x < 0 for x in img.strides):
            img = np.ascontiguousarray(img)
        shape = list(img.shape)
        shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        with torch.no_grad():
            img = torch.from_numpy(img).to(self.device)
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            img = F.interpolate(img, (new_h, new_w), mode="nearest", align_corners=None)
            shape[:2] = (new_h, new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).cpu().numpy()  # nchw -> hw(c)
        return ret

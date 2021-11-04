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


class ResizeTransformTorch(T.Transform):
    def __init__(self, h, w, new_h, new_w):
        super().__init__()
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w

    def apply_image(self, img: torch.Tensor, mode="bilinear"):
        assert len(img.shape) == 3  # C x H x W
        assert tuple(img.shape[-2:]) == (self.h, self.w)
        with torch.no_grad():
            img = F.interpolate(img.float()[None], (self.new_h, self.new_w), mode=mode)[
                0
            ]
        return img

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def inverse(self):
        return ResizeTransformTorch(self.new_h, self.new_w, self.h, self.w)

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        return self.apply_image(segmentation, mode="nearest")


class HFlipTransformTorch(T.Transform):
    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 3  # C x H x W
        return torch.flip(img, dims=[2])

    def apply_coords(self, coords):
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords

    def inverse(self) -> T.Transform:
        """
        The inverse is to flip again
        """
        return self

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

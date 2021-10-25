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

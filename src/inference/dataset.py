from pathlib import Path
from typing import List

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Dt2InferenceDataset(Dataset):
    def __init__(
        self, image_paths: List[Path], min_sizes: List[int], max_sizes: List[int]
    ):
        self.image_paths = image_paths
        self.augs = {
            (min_size, max_size): T.ResizeShortestEdge([min_size, min_size], max_size)
            for min_size, max_size in zip(min_sizes, max_sizes)
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        original_image = cv2.imread(str(path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        height, width = original_image.shape[:2]
        inputs = {
            "image": torch.from_numpy(original_image).permute(2, 0, 1),
            "height": height,
            "width": width,
        }
        for k, aug in self.augs.items():
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs[k] = {"image": image, "height": height, "width": width}
        return inputs


def get_dataloader_for_inference(
    image_paths: List[Path],
    min_sizes: List[int],
    max_sizes: List[int],
    batch_size: int,
    num_workers: int,
):
    dataset = Dt2InferenceDataset(image_paths, min_sizes, max_sizes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )
    return dataloader

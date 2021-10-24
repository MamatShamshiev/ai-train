from pathlib import Path

import cv2
import numpy as np
from defs import PROCESSED_MASKS_PATH, RAW_MASKS_PATH
from tqdm import tqdm


def process_raw_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Remaps classes from (6, 7, 10) to (1, 2, 3)

    Args:
        raw_mask (np.ndarray): raw mask

    Returns:
        np.ndarray: processed mask
    """
    for idx, v in enumerate((0, 6, 7, 10)):
        raw_mask[raw_mask == v] = idx
    return raw_mask


def read_mask(path: Path, process: bool = False) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if process is True:
        mask = process_raw_mask(mask)
    return mask


def process_for_dt2_visualization(mask):
    _m = mask == 0
    mask[_m] = 255
    mask[~_m] -= 1
    return mask


def main():
    PROCESSED_MASKS_PATH.mkdir(parents=True, exist_ok=True)
    for path_to_mask in tqdm(RAW_MASKS_PATH.glob("*.png")):
        mask = read_mask(path_to_mask, process=True)
        cv2.imwrite(str(PROCESSED_MASKS_PATH / path_to_mask.name), mask)


if __name__ == "__main__":
    main()

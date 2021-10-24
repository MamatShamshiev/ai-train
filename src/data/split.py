import json
import os

from defs import (
    PROCESSED_DATA_PATH,
    PROCESSED_MASKS_PATH,
    RAW_DATA_PATH,
    RAW_IMAGES_PATH,
    TRAIN_DATA_PATH,
    TRAIN_IMAGES_PATH,
    TRAIN_MASKS_PATH,
    VAL_DATA_PATH,
    VAL_IMAGES_PATH,
    VAL_MASKS_PATH,
)
from sklearn.model_selection import train_test_split


def make_split(test_size: float = 0.05):
    with open(RAW_DATA_PATH / "detection_coco.json", "r") as f:
        coco = json.load(f)

    filenames = [item["file_name"] for item in coco["images"]]
    _, test_filenames = train_test_split(filenames, test_size=test_size)
    with open(PROCESSED_DATA_PATH / "test_filenames.json", "w") as f:
        json.dump(test_filenames, f)


def main():
    if not (PROCESSED_DATA_PATH / "test_filenames.json").exists():
        make_split()

    with open(PROCESSED_DATA_PATH / "test_filenames.json", "r") as f:
        test_filenames = json.load(f)
    with open(RAW_DATA_PATH / "detection_coco.json", "r") as f:
        all_annotations = json.load(f)

    train_annotations = {
        k: v for k, v in all_annotations.items() if k not in ["annotations", "images"]
    }
    val_annotations = {
        k: v for k, v in all_annotations.items() if k not in ["annotations", "images"]
    }
    train_annotations["images"] = [
        im for im in all_annotations["images"] if im["file_name"] not in test_filenames
    ]
    val_annotations["images"] = [
        im for im in all_annotations["images"] if im["file_name"] in test_filenames
    ]
    train_annotations["annotations"] = [
        ann
        for ann in all_annotations["annotations"]
        if ann["file_name"] not in test_filenames
    ]
    val_annotations["annotations"] = [
        ann
        for ann in all_annotations["annotations"]
        if ann["file_name"] in test_filenames
    ]
    for path in [TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, VAL_IMAGES_PATH, VAL_MASKS_PATH]:
        path.mkdir(parents=True, exist_ok=True)
    for annotations, path in zip(
        [train_annotations, val_annotations], [TRAIN_DATA_PATH, VAL_DATA_PATH]
    ):
        with open(path / "coco.json", "w") as f:
            json.dump(annotations, f)

    for img_path in RAW_IMAGES_PATH.glob("*.png"):
        if img_path.name not in test_filenames:
            images_path = TRAIN_IMAGES_PATH
            masks_path = TRAIN_MASKS_PATH
        else:
            images_path = VAL_IMAGES_PATH
            masks_path = VAL_MASKS_PATH
        img_out_path = images_path / img_path.name
        os.symlink(img_path, img_out_path)
        mask_path = PROCESSED_MASKS_PATH / img_path.name
        mask_out_path = masks_path / img_path.name
        os.symlink(mask_path, mask_out_path)


if __name__ == "__main__":
    main()

from pathlib import Path

ROOT = Path(__file__).absolute().parent.parent
BASELINE_ROOT = ROOT / "src" / "baseline"
RAW_DATA_PATH = ROOT / "data" / "raw"
RAW_IMAGES_PATH = RAW_DATA_PATH / "images"
RAW_MASKS_PATH = RAW_DATA_PATH / "masks"

PROCESSED_DATA_PATH = ROOT / "data" / "processed"
PROCESSED_MASKS_PATH = PROCESSED_DATA_PATH / "masks"

TRAIN_DATA_PATH = PROCESSED_DATA_PATH / "train"
TRAIN_IMAGES_PATH = TRAIN_DATA_PATH / "images"
TRAIN_MASKS_PATH = TRAIN_DATA_PATH / "masks"

VAL_DATA_PATH = PROCESSED_DATA_PATH / "val"
VAL_IMAGES_PATH = VAL_DATA_PATH / "images"
VAL_MASKS_PATH = VAL_DATA_PATH / "masks"

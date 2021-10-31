import os
import time
from pathlib import Path

from inference.submit import prepare_submit

BASE_DIR = "/home/jovyan"

PATH_TO_TEST_IMAGES = Path(os.path.join(BASE_DIR, "input/images/"))
PATH_TO_PRED = Path(os.path.join(BASE_DIR, "output/"))

EXP_NAMES = [
    "cascade-R50-2fc-FrozenBN-bs=6",
    "FPN-res101-bs=6-multiscale",
    "retina-R50-1280-1794-4096-bs=6",
]
EXP_DIRS = [Path(BASE_DIR) / "outputs" / exp_name for exp_name in EXP_NAMES]
SCORE_THRESHES = [0.0001] * len(EXP_DIRS)
NMS_THRESHES = [0.5] * len(EXP_DIRS)
ENSEMBLE_METHOD = "wbf"
ENSEMBLE_METHOD_KWARGS = {"iou_thr": 0.6}

if __name__ == "__main__":
    t1 = time.time()
    if not os.path.exists(PATH_TO_PRED):
        os.mkdir(PATH_TO_PRED)
    prepare_submit(
        PATH_TO_TEST_IMAGES,
        PATH_TO_PRED,
        EXP_DIRS,
        SCORE_THRESHES,
        NMS_THRESHES,
        ENSEMBLE_METHOD,
        **ENSEMBLE_METHOD_KWARGS
    )
    t2 = time.time()
    print(t2 - t1, "ok")

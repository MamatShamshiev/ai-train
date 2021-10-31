import os
import time
from pathlib import Path

import dt2.modeling  # noqa

from inference.submit import prepare_submit

BASE_DIR = "/home/jovyan"

PATH_TO_TEST_IMAGES = Path(
    os.path.join(BASE_DIR, "input/images/")
)  # Path("/workspace/ai-train/data/processed/val/images")
PATH_TO_PRED = Path(
    os.path.join(BASE_DIR, "output/")
)  # Path("/workspace/ai-train/src/inference/output")

SCORE_THRESH = 0.05
NMS_THRESH = 0.5

if __name__ == "__main__":
    t1 = time.time()
    if not os.path.exists(PATH_TO_PRED):
        os.mkdir(PATH_TO_PRED)
    prepare_submit(
        PATH_TO_TEST_IMAGES,
        PATH_TO_PRED,
        Path(__file__).absolute().parent / "models",
        SCORE_THRESH,
        NMS_THRESH,
    )
    t2 = time.time()
    print(t2 - t1, "ok")

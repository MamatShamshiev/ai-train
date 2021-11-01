from pathlib import Path

import torch
import yaml
from baseline.detection.yolov5.models.experimental import attempt_load
from baseline.detection.yolov5.utils.augmentations import letterbox
from baseline.detection.yolov5.utils.general import non_max_suppression, scale_coords
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances


def preprocess(orig_img, img_size, device):
    img = letterbox(orig_img, img_size, auto=False)[0]
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).float().to(device)
    img_tensor /= 255
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, list(orig_img.shape), list(img_tensor.shape)


def postprocess(
    out,
    tensor_shape,
    orig_shape,
    score_thresh,
    nms_thresh,
):
    out = non_max_suppression(out, score_thresh, nms_thresh)
    assert len(out) == 1
    pred = out[0]
    res = {}
    pred[:, :4] = scale_coords(tensor_shape[2:], pred[:, :4], orig_shape).round()
    instances = Instances(
        image_size=orig_shape[:2],
        pred_boxes=Boxes(torch.as_tensor(pred[:, :4])),
        scores=torch.as_tensor(pred[:, 4]),
        pred_classes=torch.as_tensor(pred[:, 5]),
    )
    res = {"instances": instances}
    return res


def get_img_predict(
    model, orig_img, img_size, score_thresh, nms_thresh, device=torch.device("cuda:0")
):
    img_tensor, orig_shape, tensor_shape = preprocess(orig_img, img_size, device=device)
    with torch.no_grad():
        out, _ = model(img_tensor, augment=True)
    img_predict = postprocess(out, tensor_shape, orig_shape, score_thresh, nms_thresh)
    return img_predict


def get_yolo_model(path_to_weights, device=torch.device("cuda:0")):
    detect_model = attempt_load(path_to_weights, map_location=device)
    detect_model.eval()
    return detect_model


def get_model_dict_yolo(
    exp_dir: Path,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
):
    model = get_yolo_model(exp_dir / "weights" / "best.pt")
    with open(exp_dir / "opt.yaml", errors="ignore") as f:
        cfg = yaml.safe_load(f)
    img_size = cfg["imgsz"]
    return {
        "model": model,
        "img_size": img_size,
        "score_thresh": score_thresh,
        "nms_thresh": nms_thresh,
    }

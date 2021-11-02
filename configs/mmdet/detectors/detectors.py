_base_ = [
    "../_base_/models/detectors_cascade_rcnn_r50.py",
    "../_base_/datasets/aitrain.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

data = dict(samples_per_gpu=4)
optimizer = dict(lr=0.001)
load_from = "https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth"

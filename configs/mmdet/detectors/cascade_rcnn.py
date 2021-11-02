_base_ = [
    "../_base_/models/cascade_rcnn_r50_fpn.py",
    "../_base_/datasets/aitrain.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

data = dict(samples_per_gpu=1)
optimizer = dict(lr=0.001)
load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth"

_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    # "../_base_/models/cascade_rcnn_r50_fpn.py",
    "../_base_/datasets/aitrain.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=11)))

data = dict(samples_per_gpu=2)
optimizer = dict(lr=0.001)
load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth"

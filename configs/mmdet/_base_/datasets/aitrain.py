dataset_type = "CocoDataset"
data_root = "/workspace/ai-train/data/processed/"
classes = (
    "Car",
    "Human",
    "Wagon",
    "FacingSwitchL",
    "FacingSwitchR",
    "FacingSwitchNV",
    "TrailingSwitchL",
    "TrailingSwitchR",
    "TrailingSwitchNV",
    "SignalE",
    "SignalF",
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

albu_train_transforms = [
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="HueSaturationValue",
                hue_shift_limit=10,
                sat_shift_limit=35,
                val_shift_limit=25,
            ),
            dict(type="RandomGamma"),
            dict(type="CLAHE"),
        ],
        p=0.2,
    ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=0.25,
        contrast_limit=0.25,
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="MotionBlur"),
            dict(type="GaussNoise"),
            dict(type="ImageCompression", quality_lower=75),
        ],
        p=0.1,
    ),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        keymap=dict(img="image", gt_bboxes="bboxes"),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
    ),
    dict(
        type="Resize",
        img_scale=[(3072, 1024 + 64 * i) for i in range(1)],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(3072, 1536),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "train/coco.json",
        img_prefix=data_root + "train/images/",
        pipeline=train_pipeline,
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val/coco.json",
        img_prefix=data_root + "val/images/",
        pipeline=val_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "val/coco.json",
        img_prefix=data_root + "val/images/",
        pipeline=val_pipeline,
        classes=classes,
    ),
)
evaluation = dict(interval=1, metric="bbox")

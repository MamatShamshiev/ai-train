from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg
from detectron2.data.datasets.coco_panoptic import merge_to_panoptic

from dt2.visualize import STUFF_COLORS, THING_COLORS


def register_my_dataset(
    name,
    metadata,
    image_root,
    panoptic_root,
    sem_seg_root,
    instances_json,
):
    DatasetCatalog.register(
        name,
        lambda: merge_to_panoptic(
            load_coco_json(instances_json, image_root, name),
            load_sem_seg(sem_seg_root, image_root, image_ext="png"),
        ),
    )
    MetadataCatalog.get(name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json="/tmp/placeholder/",
        sem_seg_root=sem_seg_root,
        json_file=instances_json,
        evaluator_type="coco",
        ignore_label=255,
        thing_colors=THING_COLORS,
        stuff_colors=STUFF_COLORS,
        stuff_classes=["AlternativeRailPolygon", "MainRailPolygon", "Train"],
        **metadata,
    )

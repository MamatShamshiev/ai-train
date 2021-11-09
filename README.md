<div align="center">

# :train2: 2nd Place Solution of [AI Journey Contest 2021: AITrain](https://dsworks.ru/champs/70c778a8-94e9-46de-8bad-aa2c83a757ae) :train2:

![](https://user-images.githubusercontent.com/31704546/140525116-0f7710dc-8dfe-4e0d-a8eb-c21420e79c3c.png)

</div>

## Competition description

The goal of the competition is to create a computer vision system for Semantic Rail Scene Understanding.
Developing an accurate and robust algorithm is a clear way to enhance rail traffic safety.
Successful models can be incorporated in real-time applications to warn train drivers about possible collisions with potentially hazardous objects.

The dataset consists of over 7000 images from the ego-perspective of trains.
Each image is annotated with bounding boxes of 11 different types of objects (such as `car`, `human`, `wagon`, `trailing switch`) and dense pixel-wise semantic labeling for 3 different classes.

The quality metric of the competition is weighted average of [mAP@.5](https://cocodataset.org/#detection-eval) and [meanIoU](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU):
```python
competition_metric = 0.7 * mAP@.5 + 0.3 * meanIoU
```

This is a code competition so the testing time and resources are limited:
- Time for inference: 15min for 300 images;
- 1 GPU Tesla V100 32 Gb;
- 3 vCPU;
- 94 GB RAM.

Solutions are run in Docker container in the offline mode.


# Solution

Two main architectures of the solution are [Panoptic FPN](https://arxiv.org/pdf/1901.02446.pdf) and [YOLOv5](https://github.com/ultralytics/yolov5).
We don't train separate models for semantic segmentation task but solely rely on `Panoptic FPN` and multitask learning.

In a nutshell, `Panoptic FPN` is an extended version of `Mask-RCNN` with an additional semantic segmentation branch:

<div align="center">

| ![](https://user-images.githubusercontent.com/31704546/140549903-f64819f9-8875-4673-8f87-ad2e82efdf5b.png) |
|:--:|
| *Panoptic FPN architecture. [Image source](https://arxiv.org/pdf/1901.02446.pdf)* |

</div>

`YOLOv5` is a high-performing, lightweight and very popular object detection framework.
A simple codebase allows to quickly train a model on a custom dataset making `YOLOv5` an attractive choice for CV competitions.

The solution is an ensemble of 6 models:
- `Panoptic FPN` with `ResNet101` backbone and standard `Faster-RCNN` ROI head. The shortest image side size is chosen from `[1024, 1536]` with a step of `64`.
- `Panoptic FPN` with `ResNet50` backbone and `Cascade-RCNN` ROI head. Image size: `[1024, 1536]` with a step of `64`.
- `RetinaNet` with `ResNet50` backbone. Image size: `[1280, 1796]` with a step of `64`.
- `YOLOv5m6` with `2048` image size.
- `YOLOv5m6` with `2560` image size.
- `YOLOv5l6` with `1536` image resolution and label smoothing of `0.1`.

To ensemble different models we use [Weighted Boxes Fusion (WBF)](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) for object detection and a simple average for semantic segmentation.
We also tried NMS, Soft-NMS and Non-maximum weighted but WBF demonstrated a superior performance.
We set `iou_threshold=0.6` and equal weights to all models.

<div align="center">

| ![image](https://user-images.githubusercontent.com/31704546/140558929-4a303db8-0f1c-4d96-803d-7ce172593940.png) |
|:--:|
| *Weighted Boxes Fusion. [Image source](https://arxiv.org/pdf/1910.13302.pdf)* |

</div>

A bag tricks, tweaks and freebies is used to improve the performance:
- Multitask learning: all Detectron2 models are trained to solve both object detection and semantic segmentation tasks.
Multitask learning, if applied correctly, improves generalization and reduces overfiting.
Moreover, solving both tasks at once makes an inference more efficient.
- Test time augmentations: for each model run an inference on several augmented versions of original images.
We use image resizing augmentation with `[0.8, 1, 1.2]` scales with respect to maximum training image size.
- High image resolution on both training and inference.
The dataset contains quite a large amount of tiny objects so it is crucial to use high resolution images.
- Multi-scale training: using different image resolutions during training enhances the final performace.
- Light augmentations: the list of used augmentations is limited to only Random Crop, Random Brightness, Random Contrast and Random Saturation.
The flips are not used since there are some classes that depend on the sides (e.g. `facing switch left` or `facing switch right`)
Harder color and spatial augmentations hurt the performance probably due to the vast amount of tiny objects and objects which class is might be recognized only by object's color (e.g. traffic light permitting or not).


The implementation is heavily based on [Detectron2](https://github.com/facebookresearch/detectron2) and [YOLOv5](https://github.com/ultralytics/yolov5) frameworks.

<div align="center">
<a href="https://github.com/facebookresearch/detectron2"><img alt="Detectron2" src="https://raw.githubusercontent.com/facebookresearch/detectron2/main/.github/Detectron2-Logo-Horz.svg" width=384></a>
<a href="https://github.com/ultralytics/yolov5"><img alt="YOLOv5" src="https://user-images.githubusercontent.com/31704546/140525752-c24d1207-2c0c-4f8f-b050-51745af71b9f.jpg" width=384></a>
</div>


## Results
The results in the table correspond to an inference without TTA if not specified otherwise.

<div align="center">

|  Run № | Model                                                              | mAP:0.5 local | mIoU local | Metric local | mAP:0.5 public LB | mIoU public LB | Metric public LB |
|:---:   | :---:                                                              |     :---:     |    :---:   |    :---:     |       :---:       |    :---:       |    :---:         |
|   1    | `Panoptic FPN, ResNet50`                                           | 0.583         | 0.8778     |0.6716        |0.375              |0.892           |0.530             |
|   2    | `Panoptic FPN, ResNet101`                                          | 0.604         | 0.8885     |0.6893        |&mdash;            |&mdash;         |&mdash;           |
|   4    | `Panoptic FPN, ResNet50, Cascade ROI head`                         | 0.606         | 0.8626     |0.6832        |&mdash;            |&mdash;         |&mdash;           |
|   5    | `RetinaNet, ResNet50`                                              | 0.594         | &mdash;    |&mdash;       |&mdash;            |&mdash;         |&mdash;           |
|   6    | `YOLOv5m6, TTA, img_size=2048`                                     | 0.619         | &mdash;    |&mdash;       |&mdash;            |&mdash;         |&mdash;           |
|   7    | `YOLOv5m6, TTA, img_size=2560`                                     | 0.606         | &mdash;    |&mdash;       |&mdash;            |&mdash;         |&mdash;           |
|   9    | `YOLOv5l6, TTA, img_size=1536, label_smoothing=0.1`                | 0.607         | &mdash;    |&mdash;       |&mdash;            |&mdash;         |&mdash;           |
|        | Ensembled run numbers                                              |
|        | `2 + 4 + 5`                                                        | 0.642         | 0.8855     |0.7153        |0.415              |0.897           |0.560             |
|        | `2 + 4 + 6`                                                        | 0.657         | 0.8855     |0.7255        |&mdash;            |&mdash;         |&mdash;           |
|        | `2 + 4 + 5 + 6`                                                    | 0.669         | 0.8855     |0.7341        |0.421              |0.897           |0.564             |
|        | `2 + 4 + 5 + 6 + 7`                                                | 0.676         | 0.8855     |0.7393        |0.440              |0.897           |0.577             |
|        | `2 + 4 + 5 + 6 + 7 with TTA`                                       | 0.667         | 0.8875     |0.7336        |<b>0.453</b>       |0.899           |<b>0.587</b>      |
|        | `2 + 4 + 5 + 6 + 7 + 9`                                            | <b>0.685</b>  | 0.8855     |<b>0.7449</b> |0.434              |0.897           |0.573             |
|        | `2 + 4 + 5 + 6 + 7 + 9 with TTA`                                   | 0.674         | 0.8875     |0.7384        |0.447              |0.899           |0.583             |
</div>


# How to run

## 🐳&nbsp;&nbsp;Docker
Start a Docker container via docker-compose:
```bash
JUPYTER_PORT=8888 GPUS=all docker-compose -p $USER up -d --build
```
All the following steps are supposed to be run in the container.

## Dataset

[Download](https://api.dsworks.ru/dsworks-champ/api/v1/file/AITrain_train.zip/download) and unpack the data into `data/raw` directory:
```
data/
├── raw/
│   ├── bboxes
│   ├── images
│   └── masks
```

Run the following commands to prepare the dataset for Detectron2 models:
```
PYTHONPATH=$(pwd)/src python3 -m data.data2coco
PYTHONPATH=$(pwd)/src/baseline python3 -m evaluation.masks2json --path_to_masks data/raw/masks --path_to_save test.json
PYTHONPATH=$(pwd)/src python3 -m data.prepare_masks
PYTHONPATH=$(pwd)/src python3 -m data.split
```
To prepare the dataset for YOLOv5 use the [baseline notebook](https://github.com/sberbank-ai/railway_infrastructure_detection_aij2021/blob/9ccef76246b3635df8479311b7a67bd6d3454161/Detection_baseline.ipynb) provided by organizers.

The data structure after the data preparation should look as following:
```
data/
├── raw/
│   ├── bboxes
│   ├── images
│   ├── masks
│   ├── detection_coco.json
│   └── segmentation_coco.json
├── processed/
│   ├── train
│   ├── val
│   ├── masks
│   └── test_filenames.json
├── yolo/
│   ├── images
|   └── labels
```
You can take a look at the processed dataset with the [visualization notebook](notebooks/EDA.ipynb).

## Training
### Detectron2 models
The configs for Detectron2 models are located [here](configs/dt2).
For example, to train a `Panoptic FPN` with `ResNet101` backbone run the following command
```bash
bash train_dt2.sh my-sota-run main-v100
```

### YOLOv5 models
To train a `YOLOv5` model run the following commands
```bash
cd src/baseline/yolov5
python3 train.py --rect --img 2048 --batch 16 --epochs 100 --data aitrain_dataset.yaml --weights yolov5m6.pt --hyp data/hyps/hyp_aitrain.yaml --name my-sota-run
```

## Evaluation
Run [this notebook](notebooks/Evaluation.ipynb) to evaluate the model and to also run a grid search for inference parameters.
To visualize and look at the predictions use [this notebook](notebooks/Inference.ipynb).

## Make a submission
The training results (model weights and configs) should be located in `outputs/` directory.
Modify the [solution file](src/inference/solution.py) to select the required runs and run
```bash
./make_submission.sh "dt2-model-1,dt2-model2,dt2-model3" "yolo-model-1,yolo-model-2"
```

## References
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [YOLOv5](https://github.com/ultralytics/yolov5) 🚀
- [Weighted boxes fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
- [Panoptic FPN](https://arxiv.org/pdf/1901.02446.pdf)
- [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726.pdf)
- [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)

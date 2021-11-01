#!/usr/bin/env bash

DT2_EXP_DIRS=$1
YOLO_EXP_DIRS=$2

rm -rf submission.zip submission/outputs submission/src
mkdir --parents submission/outputs/
cp -r src submission

IFS=,
for val in $DT2_EXP_DIRS;
do
mkdir --parents submission/outputs/$val
cp outputs/$val/config.yaml submission/outputs/$val
cp outputs/$val/model_best.pth submission/outputs/$val
done

for val in $YOLO_EXP_DIRS;
do
mkdir --parents submission/outputs/$val/weights
cp outputs/$val/opt.yaml submission/outputs/$val  # yolo config
cp outputs/$val/weights/best.pt submission/outputs/$val/weights/ # dt2 weights
done

cd submission; zip -r ../submission.zip *
#!/usr/bin/env bash

EXP_DIRS=$1

rm -r submission.zip submission/outputs submission/src
mkdir submission/outputs
cp -r src submission
IFS=,
for val in $EXP_DIRS;
do
mkdir submission/outputs/$val
cp outputs/$val/config.yaml submission/outputs/$val
cp outputs/$val/model_best.pth submission/outputs/$val
done
cd submission; zip -r ../submission.zip *
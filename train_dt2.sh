#!/bin/bash

EXP_NAME=$1
CONFIG=$2
shift 2

PYTHONPATH=$(pwd)/src python -m train_dt2 --config-file configs/dt2/$CONFIG --num-gpus 1 OUTPUT_DIR $(pwd)/outputs/$EXP_NAME "$@"

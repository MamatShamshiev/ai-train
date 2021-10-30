#!/bin/bash

EXP_NAME=$1
CONFIG=$2
shift 2

PYTHONPATH=$(pwd)/src python -m train --config-file configs/$CONFIG --num-gpus 1 OUTPUT_DIR $(pwd)/outputs/$EXP_NAME "$@"

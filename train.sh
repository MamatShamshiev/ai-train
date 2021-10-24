#!/bin/bash

EXP_NAME=$1
CONFIG="main.yaml"
shift 1

PYTHONPATH=$(pwd)/src python -m train --config-file configs/$CONFIG --num-gpus 1 OUTPUT_DIR $(pwd)/outputs/$EXP_NAME "$@"

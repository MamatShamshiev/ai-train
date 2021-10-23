#!/usr/bin/env bash

docker rm -f mamat_aitrain_1

JUPYTER_PORT=8888 GPUS=all docker-compose -p $USER up -d --build --remove-orphans

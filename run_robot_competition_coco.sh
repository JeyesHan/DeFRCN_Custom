#!/usr/bin/env bash

EXPNAME=$1
SAVEDIR=checkpoints/${EXPNAME}
IMAGENET_PRETRAIN=/data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=/data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path

BASE_WEIGHT=/home/hanj/pyprojects/DeFRCN/checkpoints/coco/defrcn/defrcn_det_r101_base/model_reset_remove.pth
CONFIG_PATH=configs/voc/robot_competition_coco.yaml
OUTPUT_DIR=${SAVEDIR}/defrcn_fsod_r101_novel/fsrw-like

python3 main.py --num-gpus 8 --config-file ${CONFIG_PATH}                  \
    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}           \
            TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

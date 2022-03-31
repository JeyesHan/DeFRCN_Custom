#!/usr/bin/env bash

EXP_NAME=$1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=/home/hanj/pyprojects/DeFRCN/datasets/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=/home/hanj/pyprojects/DeFRCN/datasets/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path


# # ------------------------------- Base Pre-train ---------------------------------- #
# python3 main.py --num-gpus 8 --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#     --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
#            OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}


# # ------------------------------ Model Preparation -------------------------------- #
# python3 tools/model_surgery.py --dataset voc --method remove                                    \
#     --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                      \
#     --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=$2
SPLIT_ID=1
# ------------------------------ Novel Fine-tuning -------------------------------- #
# --> 1. FSRW-like, i.e. run seed0 10 times (the FSOD results on voc in most papers)
for repeat_id in 0
do
    for shot in 10   # if final, 10 -> 1 2 3 5 10
    do
        for seed in 0
        do
            # python3 tools/create_config.py --dataset voc --config_root configs/voc \
            #     --shot ${shot} --seed ${seed} --setting 'fsod' --split ${SPLIT_ID}
            CONFIG_PATH=configs/voc/robot_competition.yaml
            OUTPUT_DIR=${SAVE_DIR}/defrcn_fsod_r101_novel${SPLIT_ID}/fsrw-like/${shot}shot_seed${seed}_repeat${repeat_id}
            python3 main.py --dist-url auto --num-gpus 8 --config-file ${CONFIG_PATH} --eval-only \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                   \
                       TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
            # rm ${CONFIG_PATH}
            # rm ${OUTPUT_DIR}/model_final.pth
        done
    done
done
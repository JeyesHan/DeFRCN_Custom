import torch, torchvision

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import shutil
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from defrcn.engine import DefaultPredictor
from defrcn.config import get_cfg, set_global_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file("/home/hanj/pyprojects/DeFRCN_smog/configs/voc/robot_competition.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = '/home/hanj/pyprojects/DeFRCN_smog/checkpoints/voc/robot_competition/defrcn_fsod_r101_novel/fsrw-like/10shot_seed0_repeat0/model_0000499.pth' # model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

test_dir = sys.argv[1]
out_dir = sys.argv[2]

if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
predictor = DefaultPredictor(cfg)

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    im = cv2.imread(img_path)
    
    # inference
    outputs = predictor(im)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_img_path = os.path.join(out_dir, img_name)
    cv2.imwrite(out_img_path, out.get_image()[:, :, ::-1])
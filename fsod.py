import torch, torchvision

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import shutil
import numpy
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from defrcn.engine import DefaultPredictor
from defrcn.config import get_cfg, set_global_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class FSOD:
    def __init__(self, config_file, model_weights, thresh=0.5, ):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        self.cfg.MODEL.WEIGHTS = model_weights
        self.predictor = DefaultPredictor(self.cfg)
    
    def inference(
        self,
        img: numpy.ndarray,
    ) -> dict:
        outputs = self.predictor(img)
        return outputs

if __name__ == '__main__':

    fsod_detector = FSOD(
        config_file="configs/voc/robot_competition.yaml",
        model_weights="checkpoints/voc/robot_competition_more/defrcn_fsod_r101_novel/fsrw-like/10shot_seed0_repeat0/model_final.pth",
    )
    img_path = '97562.jpg'
    im = cv2.imread(img_path)
    outputs = fsod_detector.inference(im)
    print(outputs)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(fsod_detector.cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_img_path = 'out.jpg'
    cv2.imwrite(out_img_path, out.get_image()[:, :, ::-1])

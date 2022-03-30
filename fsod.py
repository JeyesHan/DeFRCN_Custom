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
from detectron2.structures.instances import Instances

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


class FSOD_OS(FSOD):
    def __init__(self, config_file, model_weights, object='apple', return_box_conf=True, thresh=0.001):
        super().__init__(config_file, model_weights, thresh)
        self.object = object
        self.return_box_conf = return_box_conf
        self.thing_classes = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
        if self.object:
            assert self.object in self.thing_classes, 'unsupported object category to search'
            for i, cls in enumerate(self.thing_classes):
                if cls == self.object:
                    self.object_cid = i
                    break

    def inference(
        self,
        img: numpy.ndarray,
    ) -> Instances:
        outputs = self.predictor(img)
        outputs = self.object_search(outputs)
        return outputs
    
    def object_search(self, outputs):
        scores = outputs["instances"].to("cpu").scores.numpy()
        classes = outputs["instances"].to("cpu").pred_classes.numpy()
        if self.object:
            scores[np.where(classes!=self.object_cid)] = 0
            preserve_index = np.argmax(scores)
        else:
            preserve_index = np.argmax(scores)
        instances = outputs["instances"].to("cpu")
        instances.pred_boxes.tensor = instances.pred_boxes.tensor[preserve_index:preserve_index+1]
        instances.pred_classes = instances.pred_classes[preserve_index:preserve_index+1]
        instances.scores = instances.scores[preserve_index:preserve_index+1]
        if not self.return_box_conf:
            instances.remove('scores')
        return instances
    

if __name__ == '__main__':
    fsod_detector = FSOD_OS(
        config_file="configs/voc/robot_competition.yaml",
        model_weights="../robot_fsod_models/step2.pth",
        object='biscuit' # this param is very important, please set the class you want to search. choose from ['purple bottle', 'biscuit', 'apple']
    )
    img_path = '05.jpg'
    im = cv2.imread(img_path)
    outputs = fsod_detector.inference(im)
    print(outputs)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(fsod_detector.cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs)
    out_img_path = 'out.jpg'
    cv2.imwrite(out_img_path, out.get_image()[:, :, ::-1])

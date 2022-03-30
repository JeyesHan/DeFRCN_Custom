import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import sys
import shutil
import argparse
import glob
import numpy as np
import os, json, cv2, random


from detectron2 import model_zoo
from defrcn.engine import DefaultPredictor
from defrcn.config import get_cfg, set_global_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from fsod import FSOD

def instances2json(
    instances, # detectron2 Instances
    imagePath: str,
    thing_classes: list,
    save_path: str="",
    high_conf_box: bool=True,
) -> dict:
    ret = {"version": "5.0.1", "flags": {}, "shapes": [], "imageData": None,}
    ret["imagePath"] = imagePath
    ret["imageHeight"], ret["imageWidth"] = instances.image_size
    scores = instances.scores.tolist()
    max_score_index = np.argmax(scores)
    for i, (box, score, pred_cls) in enumerate(zip(instances.pred_boxes.tensor.tolist(), instances.scores.tolist(), instances.pred_classes)):
        if high_conf_box:
            if i != max_score_index:
                continue
            if pred_cls >= len(thing_classes):
                continue
        ret["shapes"].append({
            "label": thing_classes[pred_cls],
            "points": [
                [box[0], box[1]],
                [box[2], box[3]]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(ret, f, indent=1)
    return save_path
    
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/voc/robot_competition.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument(
    #     "--input",
    #     help="A list of space separated input images; "
    #     "or a single glob pattern such as 'directory/*.jpg'",
    # )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--weights",
        help="model weights to load",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.01,
        help="Minimum score for instance predictions to be shown",
    )
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    fsod_detector = FSOD(
        config_file=args.config_file,
        model_weights=args.weights,
        thresh=args.confidence_threshold
    )
    image_dir = '/home/hanj/pyprojects/robot_initial/labelme_images/no_label_images'
    save_dir = args.output # '/home/hanj/pyprojects/robot_initial/vis_step0'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_paths.sort()
    os.system('rm -rf /home/hanj/pyprojects/robot_initial/labelme_images/no_label_images/*.json')
    for i, img_pth in enumerate(image_paths):
        if (i+1) % 10 != 0: # using features sparsely (3 frames per second)
            continue
        im = cv2.imread(img_pth)
        outputs = fsod_detector.inference(im)
        out_json_path = img_pth.replace('.jpg', '.json')
        instances2json(
            outputs["instances"].to("cpu"),
            img_pth.split('/')[-1],
            MetadataCatalog.get(fsod_detector.cfg.DATASETS.TRAIN[0]).thing_classes,
            out_json_path
        )
    os.chdir('/home/hanj/pyprojects/labelme-main/examples/bbox_detection')
    os.system('python labelme2voc.py /home/hanj/pyprojects/robot_initial/labelme_images/no_label_images tmp --labels /home/hanj/pyprojects/robot_initial/labelme_images/labels.txt')
    os.system('mv tmp/AnnotationsVisualization %s' % save_dir)
    os.system('rm -rf tmp')

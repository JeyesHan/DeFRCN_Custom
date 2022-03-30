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
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from fsod import FSOD

def get_anno(dirname, fileid, classnames):
    anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
    jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

    tree = ET.parse(anno_file)

    r = {
        "file_name": jpeg_file,
        "image_id": fileid,
        "height": int(tree.findall("./size/height")[0].text),
        "width": int(tree.findall("./size/width")[0].text),
    }
    instances = []

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        if not (cls in classnames):
            continue
        bbox = obj.find("bndbox")
        bbox = [
            float(bbox.find(x).text)
            for x in ["xmin", "ymin", "xmax", "ymax"]
        ]
        bbox[0] -= 1.0
        bbox[1] -= 1.0

        instances.append(
            {
                "category_id": classnames.index(cls),
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
            }
        )
    r["annotations"] = instances
    return r

def iou(pred_box, gt_boxes):
    ixmin = np.maximum(pred_box[:, 0], gt_boxes[0])
    iymin = np.maximum(pred_box[:, 1], gt_boxes[1])
    ixmax = np.minimum(pred_box[:, 2], gt_boxes[2])
    iymax = np.minimum(pred_box[:, 3], gt_boxes[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (gt_boxes[2] - gt_boxes[0] + 1.0) * (gt_boxes[3] - gt_boxes[1] + 1.0)
        + (pred_box[:, 2] - pred_box[:, 0] + 1.0) * (pred_box[:, 3] - pred_box[:, 1] + 1.0)
        - inters
    )

    overlaps = inters / uni
    return overlaps

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/voc/robot_competition.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
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
        default=0.001,
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
    image_dir = args.input # '/home/hanj/pyprojects/robot_initial/labelme_images/no_label_images'
    save_dir = args.output # '/home/hanj/pyprojects/robot_initial/vis_step0'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_paths.sort()
    recall_num = 0
    total_num = 0
    class_aware = True
    for img_pth in image_paths:
        # obtain gt
        fileid = img_pth.split('/')[-1].split('.')[0]
        thing_classes = MetadataCatalog.get(fsod_detector.cfg.DATASETS.TRAIN[0]).thing_classes
        GT = get_anno(os.path.dirname(image_dir), fileid, thing_classes)
        gt_box_class = GT['annotations'][0]['category_id']
        gt_box = GT['annotations'][0]['bbox']

        total_num += 1
        im = cv2.imread(img_pth)
        outputs = fsod_detector.inference(im)
        
        scores = outputs["instances"].to("cpu").scores.numpy()
        classes = outputs["instances"].to("cpu").pred_classes.numpy()
        if class_aware:
            scores[np.where(classes!=gt_box_class)] = 0
            preserve_index = np.argmax(scores)
        else:
            preserve_index = np.argmax(scores)
        instances_cpu = outputs["instances"].to("cpu")
        instances_cpu.pred_boxes.tensor = instances_cpu.pred_boxes.tensor[preserve_index:preserve_index+1]
        instances_cpu.remove('scores') # = [] # instances_cpu.scores[preserve_index:preserve_index+1]
        instances_cpu.pred_classes = instances_cpu.pred_classes[preserve_index:preserve_index+1]

        pred_boxes = instances_cpu.pred_boxes.tensor.numpy()
        pred_classes = instances_cpu.pred_classes
        

        pred_boxes_of_gt_class = pred_boxes[np.where(pred_classes==gt_box_class)]
        recall_flag = False
        if len(pred_boxes_of_gt_class) > 0:
            ious = iou(pred_boxes_of_gt_class, gt_box)
            if max(ious) >= 0.5:
                recall_num += 1
                recall_flag = True

        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(fsod_detector.cfg.DATASETS.TRAIN[0]), scale=1.2)

        out = v.draw_instance_predictions(instances_cpu)
        
        prefix = 'missed_' if not recall_flag else ''
        out_img_path = os.path.join(save_dir, prefix + img_pth.split('/')[-1])
        cv2.imwrite(out_img_path, out.get_image()[:, :, ::-1])

    print('Recall {}/{}, Score {:.3f}'.format(recall_num, total_num, recall_num / total_num))
        

import sys
import os

sys.path.append(os.getcwd())

from PIL import Image
from transformers import pipeline

import argparse

from typing import Dict


def run(conf: Dict) -> None:
    # Load a model
    checkpoint = "google/owlvit-base-patch32"
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

    # load data
    img = Image.open(conf["image_path"]).convert("RGB")

    # Run inference on '*.jpg' with arguments
    predictions = detector(
        img,
        candidate_labels=["human face", "star-spangled banner"],
    )


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="eyema_OD/weights/yolov8n.pt",
        choices=[
            "eyema_OD/weights/yolov8n.pt",
            "eyema_OD/weights/yolov8s.pt",
            "eyema_OD/weights/yolov8m.pt",
            "eyema_OD/weights/yolov8l.pt",
            "eyema_OD/weights/yolov8x.pt",
        ],
        help="yolo model options",
    )

    # more informations: https://docs.ultralytics.com/modes/predict/#__tabbed_2_4
    parser.add_argument(
        "--data",
        type=str,
        default="/Users/miladsoleymani/Desktop/eyema_OD/cfg/train.yaml",
        help="path to data file, i.e. coco128.yaml",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train for",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="size of input images as integer",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)

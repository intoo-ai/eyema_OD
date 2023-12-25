# requierment: pip install -q transformers

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
    img = Image.open(conf["data_path"]).convert("RGB")

    # Split the input string using the comma as the delimiter
    candidate_labels = conf["candidate_labels"].split(", ")

    # Run inference on '*.jpg' with arguments
    predictions = detector(
        img,
        candidate_labels=candidate_labels,
    )

    print(predictions)


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/tesla/images/imgs/frame_0089.jpg",
        help="path to data file, i.e. img.jpg",
    )

    parser.add_argument(
        "--candidate_labels",
        type=str,
        default="",
        help="example: human face, star-spangled banner",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)

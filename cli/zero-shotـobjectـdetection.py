# requierment: 1 pip install -q transformers
#              2 pip install -q tensorrt

import sys
import os

sys.path.append(os.getcwd())

from PIL import Image
from PIL import ImageDraw

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

    draw = ImageDraw.Draw(img)

    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

    img.save(conf["save_path"])


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/eyema_OD/sampl_data/000000000025.jpg",
        help="path to data file, i.e. img.jpg",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="/content/results.jpg",
        help="path to save data file, i.e. img.jpg",
    )

    parser.add_argument(
        "--candidate_labels",
        type=str,
        default="animal",
        help="example: human face, star-spangled banner",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)

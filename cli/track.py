import sys
import os

sys.path.append(os.getcwd())

from ultralytics import YOLO

import argparse

from typing import Dict


def run(conf: Dict) -> None:
    class_names = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    # Creating a dictionary with names as keys and indices as values
    class_indices = {class_name: index for index, class_name in enumerate(class_names)}

    classes = list(class_indices.values())

    if conf["classes"] is not None:
        # Convert the string to a list using split
        word_list = conf["classes"].split()
        # Using map() function
        classes = list(map(class_indices.get, word_list))

    # Load a model
    model = YOLO(conf["model"])  # load an official model

    # Run inference on 'bus.jpg' with arguments
    model.track(
        source=conf["source"],
        conf=conf["conf"],
        iou=conf["iou"],
        device=conf["device"],
        classes=classes,
        save=conf["save"],
        save_frames=conf["save_frames"],
        save_txt=conf["save_txt"],
        save_conf=conf["save_conf"],
        show_labels=conf["show_labels"],
        show_boxes=conf["show_boxes"],
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
        "--source",
        type=str,
        default="test2.mp4",
        help="source directory for videos: video.mp4",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="object confidence threshold for detection",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="intersection over union (IoU) threshold for NMS",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to run on, i.e. cuda device=0/1/2/3 or device=cpu",
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="a string contain names of objects with space, example: couch potted plant bed ",
    )

    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="show predicted images and videos if environment allows",
    )

    parser.add_argument(
        "--save_frames",
        type=bool,
        default=False,
        help="save predicted individual video frames",
    )

    parser.add_argument(
        "--save_txt",
        type=bool,
        default=False,
        help="save results as .txt file",
    )

    parser.add_argument(
        "--save_conf",
        type=bool,
        default=True,
        help="show prediction confidence, i.e. '0.99'",
    )

    parser.add_argument(
        "--show_labels",
        type=bool,
        default=True,
        help="show prediction labels, i.e. 'person'",
    )

    parser.add_argument(
        "--show_boxes",
        type=bool,
        default=True,
        help="show prediction boxes",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)

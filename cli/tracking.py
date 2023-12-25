import json
import argparse
from typing import Dict


def create_intermediate_label(label1, label2, t):
    # Interpolate between two labels based on time parameter (t)
    class_id = label1[0]
    height = label1[1] + t * (label2[1] - label1[1])
    width = label1[2] + t * (label2[2] - label1[2])
    x_center = label1[3] + t * (label2[3] - label1[3])
    y_center = label1[4] + t * (label2[4] - label1[4])

    return class_id, height, width, x_center, y_center


def follow_objects_between_frames(
    first_frame, last_frame, first_frame_number, last_frame_number
):
    # Initialize an empty list to store intermediate labels
    intermediate_labels = []

    # Generate intermediate labels for each frame
    for frame in range(first_frame_number, last_frame_number):
        # Calculate the time parameter (t) for interpolation
        t = (frame - first_frame_number) / (last_frame_number - first_frame_number)

        # Create intermediate label for the current frame
        intermediate_label = create_intermediate_label(first_frame, last_frame, t)

        # Append the intermediate label to the list
        intermediate_labels.append(intermediate_label)

    return intermediate_labels


def run(conf: Dict) -> None:
    # Specify the path to your JSON file
    json_file_path = conf["json_path"]  # or api

    # Open the file for reading
    with open(json_file_path, "r") as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)

    # Now 'data' contains the contents of the JSON file as a Python dictionary
    # You can access the data as needed
    AnnotationFrames = data["AnnotationFrames"]
    temp_first_frame = data["AnnotationFrames"][0]

    ClassId = data["AnnotationFrames"][0]["AnnotationClasses"][0]["BoundingBox"][
        "ClassId"
    ]
    points = data["AnnotationFrames"][0]["AnnotationClasses"][0]["BoundingBox"][
        "Points"
    ]
    first_frame_number = data["AnnotationFrames"][0]["FrameNumber"]
    first_frame = [ClassId, points["h"], points["w"], points["x"], points["y"]]

    ClassId = data["AnnotationFrames"][1]["AnnotationClasses"][0]["BoundingBox"][
        "ClassId"
    ]
    points = data["AnnotationFrames"][1]["AnnotationClasses"][0]["BoundingBox"][
        "Points"
    ]
    last_frame_number = data["AnnotationFrames"][1]["FrameNumber"]
    last_frame = [ClassId, points["h"], points["w"], points["x"], points["y"]]

    intermediate_labels = follow_objects_between_frames(
        first_frame, last_frame, first_frame_number, last_frame_number
    )

    # Using enumerate with a custom start value (start=1)
    for index, value in enumerate(intermediate_labels, start=first_frame_number + 1):
        temp_label = temp_first_frame
        temp_label["FrameNumber"] = index
        temp_label["AnnotationClasses"][0]["BoundingBox"]["Area"] = value[1] * value[2]
        temp_label["AnnotationClasses"][0]["BoundingBox"]["Points"] = value[1:]

        # Inserting an element (e.g., 99) between elements 2 and 3
        AnnotationFrames.insert(index, temp_label)

    data["AnnotationFrames"] = AnnotationFrames

    # Open the file for writing and save the JSON data
    with open(conf["json_save_path"], "w") as json_file:
        json.dump(data, json_file)  # indent parameter for pretty formatting


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_path",
        type=str,
        default="/Users/miladsoleymani/Desktop/eyema_OD/api/get.json",
        help="path to json file, i.e. get.json",
    )

    parser.add_argument(
        "--json_save_path",
        type=str,
        default="/Users/miladsoleymani/Desktop/eyema_OD/api/post.json",
        help="path to json file, i.e. post.json",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)

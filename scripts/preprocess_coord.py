import argparse
import collections
import torch
import json


def filter_keypoint(keypoints):
    return keypoints

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type=str, help=".")
    parser.add_argument("--output_path", type=str, help=".")

    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    data['keypoints'] =

    for l in data['keypoints']:




if __name__ == "__main__":
    main()

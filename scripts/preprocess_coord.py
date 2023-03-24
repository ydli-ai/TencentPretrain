import argparse
import collections
import torch
import json
import numpy as np

# [Annotation type] [Object centric or Multiple instances] [Number of instances] [Number of keypoints] [Class A, Class B, ...] [Box A, Box B, ...]


def filter_keypoint(keypoints):
    output = []
    for kp_list in keypoints:
        output_single = []
        for kp in kp_list:
            for name, point in kp.items():
                if np.array(point).sum() > 0:
                    output_single.append({name: point})
        if len(output_single) > 0:
            output.append(output_single)
    return output


def keypoint_to_formular_data(keypoints):
    output = []
    for kp_list in keypoints:
        output_single = {"anno_type": "key point",
                         "prefix": "Multiple instances",
                         "instances_num": len(kp_list),
                         "keypoints_num": None,
                         "categories": [],
                         "coordinate": []
                         }
        for kp in kp_list:
            for name, point in kp.items():
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
                output_single["keypoints_num"]= len(point)
        output.append(output_single)
    return output


def mask_to_formular_data(keypoints):
    output = []
    for mask_list in keypoints:
        output_single = {"anno_type": "mask",
                         "prefix": "Multiple instances",
                         "instances_num": len(mask_list),
                         "keypoints_num": 0,
                         "categories": [],
                         "coordinate": []
                         }
        for mask in mask_list:
            for name, point in mask.items():
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
        output.append(output_single)
    return output


def box_to_formular_data(keypoints):
    output = []
    for mask_list in keypoints:
        output_single = {"anno_type": "box",
                         "prefix": "Multiple instances",
                         "instances_num": len(mask_list),
                         "keypoints_num": 0,
                         "categories": [],
                         "coordinate": []
                         }
        for mask in mask_list:
            for name, point in mask.items():
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
        output.append(output_single)
    return output

num2char = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
            10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r' }

def formular_data_to_str(data_list, type):

    def keyporint_coord_to_str(keypoints):
        output = ""
        for points_list in keypoints:
            output = output + '['
            for i, point in enumerate(points_list):
                output = output + ' ' + num2char[i] + ' $' + str(point[0]) + ' $'+ str(point[0])
            output = output + '] '
        return output

    def mask_coord_to_str(keypoints):
        output = ""
        for points_list in keypoints:
            output = output + '['
            for i, point in enumerate(points_list):
                output = output + ' ' + num2char[i] + ' $' + str(point[0][0]) + ' $'+ str(point[0][0])
            output = output + '] '
        return output

    def box_coord_to_str(boxes):
        output = ""
        for box in boxes:
            output = output + '[  xmin $' + str(box[0]) + ' ymin $'+ str(box[1]) + \
            ' ymax $'+ str(box[2]) + ' ymax $'+ str(box[3]) +'] '
        return output

    output = []
    for data in data_list:
        output_single = '; '.join([data["anno_type"], data["prefix"], str(data["instances_num"]), str(data["keypoints_num"])])
        output_single = output_single + '; ' + ', '.join(data["categories"]) +'; '
        if type == "keypoint":
            output_single = output_single + keyporint_coord_to_str(data["coordinate"])
        elif type == "box":
            output_single = output_single + box_coord_to_str(data["coordinate"])
        else:
            output_single = output_single + mask_coord_to_str(data["coordinate"])
        output.append(output_single)

    return output

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type=str, help=".")
    parser.add_argument("--output_path", type=str, help=".")
    parser.add_argument("--data_type", type=str, help=".")

    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    if args.data_type == "keypoint":
        keypoints = filter_keypoint(data['keypoints'] )
        data_json = keypoint_to_formular_data(keypoints)
    elif args.data_type == "box":
        data_json = box_to_formular_data(data['bboxes'])
    else:
        data_json = mask_to_formular_data(data['masks'])

    data_str = formular_data_to_str(data_json, args.data_type)

    with open(args.output_path, 'w') as f:
        for l in data_str:
            f.write(l + '\n')


if __name__ == "__main__":
    main()

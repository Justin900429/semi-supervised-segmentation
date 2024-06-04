import argparse
import glob
import math
import os

import cv2
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

CLASS_LIST = [
    "road",
    "building",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "sky",
    "person",
    "car",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model")
    parser.add_argument(
        "--gt", type=str, required=True, help="Path to the ground truth"
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Path to the prediction"
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    return parser.parse_args()


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 12):
    ious = []
    for cls_ in range(1, num_classes):
        pred_inds = pred == cls_
        target_inds = target == cls_
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return ious


def main(args):
    predictions = sorted(list(glob.glob(os.path.join(args.pred, "*_mask.png"))))

    total_ious = [[] for _ in range(args.num_classes - 1)]
    for pred in tqdm(predictions):
        gt = os.path.join(args.gt, os.path.basename(pred))
        gt_image = cv2.imread(gt, 0)
        pred_image = cv2.imread(pred, 0)
        try:
            ious = compute_iou(pred_image, gt_image, args.num_classes)
        except Exception as e:
            print(str(e))
            ious = [0] * (args.num_classes - 1)

        for idx in range(args.num_classes - 1):
            if not math.isnan(ious[idx]):
                total_ious[idx].append(ious[idx])

    total_ious = [
        float("nan") if len(iou) == 0 else (sum(iou) / len(iou)) for iou in total_ious
    ]
    mean_ious = [iou for iou in total_ious if not math.isnan(iou)]
    mean_ious = sum(mean_ious) / len(mean_ious)
    header = ["class"] + CLASS_LIST + ["Average"]
    data = ["IoU"] + [f"{iou:.4f}" for iou in total_ious] + [mean_ious]
    print(tabulate([data], headers=header, tablefmt="pretty"))


if __name__ == "__main__":
    args = parse_args()
    main(args)

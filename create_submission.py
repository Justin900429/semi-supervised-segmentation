import argparse
import glob
import os

import cv2
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create submission file for gold")
    parser.add_argument("--pred", type=str, default="predictions")
    parser.add_argument("--save-file", type=str, default="submission.csv")
    return parser.parse_args()


def main(args):
    # Suppose the mask files have the name with the format as "{num}_mask.png" (recommend),
    #  otherwise, you need to modify the code below
    img_list = sorted(list(glob.glob(os.path.join(args.pred, "*_mask.png"))))
    img_list = [
        img
        for img in img_list
        if int(os.path.basename(img).replace("_mask.png", "")) > 900
    ]
    assert len(img_list) == 100

    output = []
    for img_name in img_list:
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        flatten_img = img.reshape(-1).tolist()
        str_list = [str(i) for i in flatten_img]
        output.append(
            [
                int(os.path.basename(img_name).replace("_mask.png", "")),
                " ".join(str_list),
            ]
        )

    df = pd.DataFrame(output, columns=["img_id", "label"])
    df.to_csv(args.save_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

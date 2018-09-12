from __future__ import division

import shutil
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def show():
    lbl_file = "data/0908/label_list.txt"
    lbl_dict = {}
    with open(lbl_file, "r") as fin:
        for idx, line in enumerate(fin):
            lbl_dict[str(idx)] = line.strip().split("\t")[0]

    data_dir = "data/0908/resized_images"
    infer_file = "infer_output.txt"
    out_dir = "visual_outputs"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    font = ImageFont.truetype("NotoSansCJK-Black.ttc", 12)
    path_to_im = dict()
    for line in open(infer_file):
        img_path, _, _, _ = line.strip().split("\t")
        if img_path not in path_to_im:
            im = cv2.imread(os.path.join(data_dir, img_path))
            path_to_im[img_path] = im

    for line in open(infer_file):
        img_path, label, conf, bbox = line.strip().split("\t")
        xmin, ymin, xmax, ymax = map(float, bbox.split(" "))
        xmin = int(round(xmin))
        ymin = int(round(ymin))
        xmax = int(round(xmax))
        ymax = int(round(ymax))

        img = path_to_im[img_path]

        # plot bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, (1 - xmin) * 255, xmin * 255), 2)

        # plot label
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        position = (int(round(xmin + (xmax - xmin) / 4)), ymin - 20)
        draw = ImageDraw.Draw(img_PIL)
        draw.text(
            position,
            lbl_dict[label].decode("utf8"),
            font=font,
            fill=(255, 0, 0))
        path_to_im[img_path] = cv2.cvtColor(
            np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

    for img_path in path_to_im:
        im = path_to_im[img_path]
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, im)

    print("Done.")


if __name__ == "__main__":
    show()

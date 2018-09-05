from __future__ import division

import os
import sys
import json
import requests
import codecs
from collections import defaultdict
from PIL import Image
import cv2
import pdb
import shutil


def label_dict(data_dir, save_path):
    lbl_dict = defaultdict(int)
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        label_list = json.loads(open(file_path, "r").readline())
        for lbl in label_list:
            lbl_dict[lbl["text"]] += 1

    sorted_dict = sorted(
        lbl_dict.iteritems(), key=lambda x: x[1], reverse=False)
    with codecs.open(save_path, "w", "utf8") as fout:
        for item in sorted_dict:
            fout.write("%s\t%d\n" % (item[0], item[1]))


def download_data(file_list, save_dir):
    with open(file_list, "r") as fin:
        fin.readline()
        for idx, line in enumerate(fin):
            if not idx % 50: print("download %d images." % (idx))
            url, _, name, lbl = line.strip().split()

            name = name.replace("/", "_")
            r = requests.get(url, allow_redirects=True)
            open(os.path.join(save_dir, "images", name), "wb").write(r.content)
            lbl_path = os.path.join(save_dir, "annotations",
                                    os.path.splitext(name)[0] + ".txt")
            open(lbl_path, "w").write(lbl)


def gen_train_list(data_dir, save_path):
    with open(save_path, "w") as fout:
        for img in os.listdir(os.path.join(data_dir, "images")):
            fout.write("%s %s\n" % (img, os.path.splitext(img)[0] + ".txt"))


def virtualize_bbox(image_dir, lbl_dir, save_dir, TARGET_SIZE=None):
    def __proc_one_image(image_path, lbl_path, save_path, TARGET_SIZE):
        im = cv2.imread(image_path)
        height, width, depth = im.shape

        labels = json.loads(open(lbl_path, "r").readline())
        for lbl in labels:
            xmin = int(round(lbl["posX"]))
            ymin = int(round(lbl["posY"]))
            xmax = int(round(lbl["posX"] + lbl["width"]))
            ymax = int(round(lbl["posY"] + lbl["height"]))

            if TARGET_SIZE is not None:
                im = cv2.resize(im, TARGET_SIZE)

                w_scale = TARGET_SIZE[0] / width
                h_scale = TARGET_SIZE[0] / height

                xmin = int(round(xmin * w_scale))
                ymin = int(round(ymin * h_scale))
                xmax = int(round(xmax * w_scale))
                ymax = int(round(ymax * h_scale))

            cv2.rectangle(im, (xmin, ymin), (xmax, ymax),
                          (0, (1 - xmin) * 255, xmin * 255), 2)
        cv2.imwrite(save_path, im)

    if os.path.exists(image_dir): shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for image in os.listdir(image_dir):
        __proc_one_image(
            os.path.join(image_dir, image),
            os.path.join(lbl_dir,
                         os.path.splitext(image)[0] + ".txt"),
            os.path.join(save_dir, image), TARGET_SIZE)


if __name__ == "__main__":
    # dir_name = "0905/raw_list"
    # for file_name in os.listdir(dir_name):
    #     file_path = os.path.join(dir_name, file_name)
    #     download_data(file_list=file_path, save_dir="0905")

    virtualize_bbox(
        image_dir="0905/images",
        lbl_dir="0905/annotations",
        save_dir="0905/bounding_pox",
        TARGET_SIZE=(300, 300))
    # gen_train_list("data", "data/train_list.txt")
    # label_dict("data/annotations", "data/label_list.txt")

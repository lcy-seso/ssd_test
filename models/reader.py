from __future__ import division

import os
import random
import numpy as np
import codecs
import json
import cv2
from PIL import Image
import pdb

import image_util
from paddle.utils.image_util import *


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

    if os.path.exists(image_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for image in os.listdir(image_dir):
        __proc_one_image(
            os.path.join(image_dir, image),
            os.path.join(lbl_dir,
                         os.path.splitext(image)[0] + ".txt"),
            os.path.join(save_dir, image), TARGET_SIZE)


class Settings(object):
    def __init__(self, data_dir, label_file, resize_h, resize_w, mean_value):
        self._data_dir = data_dir
        self._label_list = {}
        with codecs.open(os.path.join(data_dir, label_file), "r",
                         "utf8") as fin:
            for idx, line in enumerate(fin):
                self._label_list[line.strip().split("\t")[0]] = idx

        self._img_mean = np.array(mean_value)[:, np.newaxis,
                                              np.newaxis].astype("float32")
        self._resize_height = resize_h
        self._resize_width = resize_w

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def label_list(self):
        return self._label_list

    @property
    def resize_h(self):
        return self._resize_height

    @property
    def resize_w(self):
        return self._resize_width

    @property
    def img_mean(self):
        return self._img_mean


def get_batch_sampler():
    sampling_settings = [
        [1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0],
        [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0],
        [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0],
        [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0],
        [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0],
        [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0],
    ]

    batch_sampler = []
    for setting in sampling_settings:
        batch_sampler.append(
            image_util.sampler(
                max_sample=setting[0],
                max_trial=setting[1],
                min_scale=setting[2],
                max_scale=setting[3],
                min_aspect_ratio=setting[4],
                max_aspect_ratio=setting[5],
                min_jaccard_overlap=setting[6],
                max_jaccard_overlap=setting[7]))
    return batch_sampler


def _reader_creator(settings, file_list, mode, shuffle):
    ORIGINAL_HEIGH = 720
    ORIGINAL_WIDTH = 1280

    def reader():
        lines = open(file_list, "r").readlines()
        if shuffle:
            random.shuffle(lines)

        for line_id, line in enumerate(lines):
            if mode == "train" or mode == "test":
                img_path, label_path = line.split()
                img_path = os.path.join(settings.data_dir, "resized_images",
                                        img_path)
                label_path = os.path.join(settings.data_dir, "annotations",
                                          label_path)
            elif mode == "infer":
                img_path = os.path.join(settings.data_dir, "resized_images",
                                        line.strip().split()[0])

            img = Image.open(img_path)
            img_width, img_height = img.size
            img = np.array(img)

            if mode == "train" or mode == "test":
                labels = json.loads(open(label_path, "r").readline())
                if len(labels) == 0: continue

                bbox_labels = []
                for lbl in labels:
                    bbox_labels.append([
                        settings.label_list[lbl["text"]],  # label
                        lbl["posX"] / ORIGINAL_WIDTH,  # xmin
                        lbl["posY"] / ORIGINAL_HEIGH,  # ymin
                        (lbl["posX"] + lbl["width"]) / ORIGINAL_WIDTH,  # xmax
                        (lbl["posY"] + lbl["height"]) / ORIGINAL_HEIGH,  # ymax
                        1.  # difficult
                    ])
                sample_labels = bbox_labels

                if mode == "train":
                    sampled_bbox = image_util.generate_batch_samples(
                        get_batch_sampler(), bbox_labels, img_width,
                        img_height)

                    if len(sampled_bbox) > 0:
                        idx = int(random.uniform(0, len(sampled_bbox)))
                        img, sample_labels = image_util.crop_image(
                            img, bbox_labels, sampled_bbox[idx], img_width,
                            img_height)

            img = Image.fromarray(img)
            img = img.resize((settings.resize_w, settings.resize_h),
                             Image.ANTIALIAS)
            img = np.array(img)

            if mode == "train":
                mirror = int(random.uniform(0, 2))
                if mirror == 1:
                    img = img[:, ::-1, :]
                    for i in xrange(len(sample_labels)):
                        tmp = sample_labels[i][1]
                        sample_labels[i][1] = 1 - sample_labels[i][3]
                        sample_labels[i][3] = 1 - tmp

            img = np.transpose(img, [2, 0, 1])  # channel first
            img = img.astype("float32")
            img -= settings.img_mean
            img = img.flatten()

            if mode == "train" or mode == "test":
                if mode == "train" and len(sample_labels) == 0: continue
                yield img, sample_labels
            elif mode == "infer":
                yield img.astype("float32")

    return reader


def train(settings, file_list, shuffle=True):
    return _reader_creator(settings, file_list, "train", shuffle)


def test(settings, file_list):
    return _reader_creator(settings, file_list, "test", False)


def infer(settings, file_list):
    return _reader_creator(settings, file_list, "infer", False)


if __name__ == "__main__":
    from config.pascal_voc_conf import cfg
    mean = [57, 56, 58]

    train_file_list = "data/0908/05_train_list.txt"
    data_args = Settings(
        data_dir="data/0908",
        label_file="label_list.txt",
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=mean)

    for idx, data in enumerate(train(data_args, train_file_list)()):
        print("image %d" % (idx + 1))
        print(data)

        img = data[0].reshape(3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
        _, xmin, ymin, xmax, ymax, _ = data[1][0]

        xmin = int(round(xmin * cfg.IMG_WIDTH))
        ymin = int(round(ymin * cfg.IMG_HEIGHT))
        xmax = int(round(xmax * cfg.IMG_WIDTH))
        ymax = int(round(ymax * cfg.IMG_HEIGHT))

        # img = np.transpose(img, [1, 2, 0]).astype(np.int32)
        img = np.transpose(img, [1, 2, 0])
        cv2.imwrite("tmp.png", img)
        img = cv2.imread("tmp.png")
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, (1 - xmin) * 255, xmin * 255), 2)
        cv2.imwrite("input/img_%d.png" % (idx), img)

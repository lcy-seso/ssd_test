import os
import sys
import json
import requests
import codecs
from collections import defaultdict
import pdb


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


if __name__ == "__main__":
    dir_name = "0905/raw_list"
    for file_name in os.listdir(dir_name):
        file_path = os.path.join(dir_name, file_name)
        download_data(file_list=file_path, save_dir="0905")

    # gen_train_list("data", "data/train_list.txt")
    # label_dict("data/annotations", "data/label_list.txt")

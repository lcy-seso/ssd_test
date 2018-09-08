import os
import sys
import gzip
import numpy as np
from PIL import Image

import paddle.v2 as paddle

import reader
import vgg_ssd_net
from config.pascal_voc_conf import cfg


def _infer(inferer, infer_data, threshold):
    ret = []
    infer_res = inferer.infer(input=infer_data)
    keep_inds = np.where(infer_res[:, 2] >= threshold)[0]
    for idx in keep_inds:
        ret.append([
            infer_res[idx][0], infer_res[idx][1] - 1, infer_res[idx][2],
            infer_res[idx][3], infer_res[idx][4], infer_res[idx][5],
            infer_res[idx][6]
        ])
    return ret


def save_batch_res(ret_res, img_w, img_h, fname_list, fout):
    for det_res in ret_res:
        img_idx = int(det_res[0])
        label = int(det_res[1])
        conf_score = det_res[2]
        xmin = det_res[3] * img_w[img_idx]
        ymin = det_res[4] * img_h[img_idx]
        xmax = det_res[5] * img_w[img_idx]
        ymax = det_res[6] * img_h[img_idx]
        fout.write(fname_list[img_idx] + "\t" + str(label) + "\t" +
                   str(conf_score) + "\t" + str(xmin) + " " + str(ymin) + " " +
                   str(xmax) + " " + str(ymax))
        fout.write("\n")


def infer(eval_file_list, save_path, data_args, batch_size, model_path,
          threshold):
    detect_out = vgg_ssd_net.net_conf(mode="infer")

    assert os.path.isfile(model_path), "Invalid model."
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))

    inferer = paddle.inference.Inference(
        output_layer=detect_out, parameters=parameters)

    data_reader = reader.infer(data_args, eval_file_list)
    all_fname_list = [
        line.strip().split()[0] for line in open(eval_file_list).readlines()
    ]

    test_data = []
    fname_list = []
    img_w = []
    img_h = []
    idx = 0
    with open(save_path, "w") as fout:
        for idx, img in enumerate(data_reader()):
            if idx > 50: break
            test_data.append([img])
            fname_list.append(all_fname_list[idx])
            w, h = Image.open(
                os.path.join(data_args.data_dir, "resized_images",
                             fname_list[-1])).size
            img_w.append(w)
            img_h.append(h)
            if len(test_data) == batch_size:
                ret_res = _infer(inferer, test_data, threshold)
                save_batch_res(ret_res, img_w, img_h, fname_list, fout)
                test_data = []
                fname_list = []
                img_w = []
                img_h = []

        if len(test_data) > 0:
            ret_res = _infer(inferer, test_data, threshold)
            save_batch_res(ret_res, img_w, img_h, fname_list, fout)


if __name__ == "__main__":
    model_dir = "/data/data1/yincao/detection_data/0908/checkpoints"

    paddle.init(use_gpu=True, trainer_count=1)

    data_args = reader.Settings(
        data_dir="data/0908",
        label_file="0908_label_list.txt",
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[57, 56, 58])

    infer(
        eval_file_list="data/0908/train_list.txt",
        save_path="infer_output.txt",
        data_args=data_args,
        batch_size=4,
        model_path=(os.path.join(model_dir, "params_pass_00199.tar.gz")),
        threshold=0.3)

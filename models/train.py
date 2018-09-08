import os
import sys
import gzip

import paddle.v2 as paddle

import reader
import vgg_ssd_net
from config.pascal_voc_conf import cfg


def train(train_file_list, data_args, init_model_path, dev_file_list=None):
    optimizer = paddle.optimizer.Momentum(
        momentum=cfg.TRAIN.MOMENTUM,
        learning_rate=cfg.TRAIN.LEARNING_RATE,
        regularization=paddle.optimizer.L2Regularization(
            rate=cfg.TRAIN.L2REGULARIZATION),
        learning_rate_decay_a=cfg.TRAIN.LEARNING_RATE_DECAY_A,
        learning_rate_decay_b=cfg.TRAIN.LEARNING_RATE_DECAY_B,
        learning_rate_schedule=cfg.TRAIN.LEARNING_RATE_SCHEDULE)

    cost, detect_out = vgg_ssd_net.net_conf("train")

    parameters = paddle.parameters.create(cost)
    if init_model_path is not None:
        assert os.path.isfile(init_model_path), "Invalid model."
        parameters.init_from_tar(gzip.open(init_model_path))

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        extra_layers=[detect_out],
        update_equation=optimizer)

    feeding = {"image": 0, "bbox": 1}

    train_reader = paddle.batch(
        reader.train(data_args, train_file_list),
        batch_size=cfg.TRAIN.BATCH_SIZE)  # generate a batch image each time

    if dev_file_list is not None:
        dev_reader = paddle.batch(
            reader.test(data_args, dev_file_list),
            batch_size=cfg.TRAIN.BATCH_SIZE)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if (event.batch_id + 1) % 1 == 0:
                print("Pass %d, Batch %d, TrainCost %f, Detection mAP=%f" %
                      (event.pass_id, event.batch_id + 1, event.cost,
                       event.metrics["detection_evaluator"]))
                sys.stdout.flush()

        if isinstance(event, paddle.event.EndPass):
            if not (event.pass_id + 1) % 3:
                with gzip.open(
                        "checkpoints/params_pass_%05d.tar.gz" % event.pass_id,
                        "w") as f:
                    trainer.save_parameter_to_tar(f)

            if dev_file_list is not None:
                result = trainer.test(reader=dev_reader, feeding=feeding)
                print("Test with Pass %d, TestCost: %f, Detection mAP=%g" %
                      (event.pass_id, result.cost,
                       result.metrics["detection_evaluator"]))

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        num_passes=cfg.TRAIN.NUM_PASS,
        feeding=feeding)


if __name__ == "__main__":
    data_dir = "data/0908"
    label_file = "label_list.txt"
    init_model_path = "vgg/vgg_model.tar.gz"
    train_file_list = "data/0908/train_list.txt"
    dev_file_list = None

    paddle.init(use_gpu=True, trainer_count=1)
    data_args = reader.Settings(
        data_dir=data_dir,
        label_file=label_file,
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[57, 56, 58])
    train(
        train_file_list=train_file_list,
        dev_file_list=dev_file_list,
        data_args=data_args,
        init_model_path=init_model_path)

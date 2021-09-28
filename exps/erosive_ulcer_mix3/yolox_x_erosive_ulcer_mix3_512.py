#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 3
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.train_ann = "trainfp0803.json"
        self.val_ann = "test.json"

        self.data_num_workers = 1
        self.input_size = (512, 512)
        self.random_size = (20, 30)

        self.max_epoch = 300
        self.eval_interval = 1
        self.print_interval = 200
        self.degrees =20.0 #0.7_15 484_336

        self.test_size = (512, 512)
        self.test_conf = 0.01
        self.nmsthre = 0.1

        self.use_l1 = True

        self.mosaicp = 0.8 #0.7_20 496_341#0.6_20 480_335#0.75_20 482_337

        #self.data_dir = "/data2/qilei_chen/DATA/erosive_ulcer_mix3"
        self.data_dir = "/home/qilei/DATASETS/erosive_ulcer_mix"

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels,use_l1=self.use_l1)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            ErosiveUlcer,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetectionP,
        )

        dataset = ErosiveUlcer(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            name="images",
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=50,
            ),
        )

        dataset = MosaicDetectionP(
            dataset,
            mosaic=not no_aug,
            possibility=self.mosaicp,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=120,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import ErosiveUlcer, ValTransform

        valdataset = ErosiveUlcer(
            data_dir=self.data_dir ,
            json_file=self.val_ann,
            name="images",
            img_size=self.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
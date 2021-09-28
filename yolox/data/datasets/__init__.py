#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset, ErosiveUlcer
from .coco_classes import COCO_CLASSES,Erosive_Ulcer,Erosive_Ulcer3
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection,MosaicDetectionP
from .voc import VOCDetection
#from .erosive_ulcer_mix import ErosiveUlcer

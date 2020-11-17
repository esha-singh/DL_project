# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:36:41 2020

@author: tedkuo
"""

from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = "cuda:0"
_C.NUM_WORKERS = 0

_C.MODEL = CN()
_C.MODEL.NETWORK = "resnet50"

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.EPOCHS = 25
_C.TRAIN.SAVE_FREQUENCY = 100

_C.TRAIN.OPTIM = CN()
_C.TRAIN.OPTIM.OPTIMIZER = "adam" # "sgd", "adam", "rmsprop"
_C.TRAIN.OPTIM.BASE_LR = 0.001
_C.TRAIN.OPTIM.SGD_MOMENTUM = 0.9
_C.TRAIN.OPTIM.ADAM_BETAS = (0.9, 0.999)
_C.TRAIN.OPTIM.LR_DECAY_RATE = 0.9
_C.TRAIN.OPTIM.LR_DECAY_FREQUENCY = 50000

_C.EVAL = CN()
_C.EVAL.MODEL_PATH = ""
_C.EVAL.FIRST_SUBJECT = 101
_C.EVAL.NUM_SUBJECTS = 10
_C.EVAL.BATCH_SIZE = 8
_C.EVAL.SAVE_FREQUENCY = 5
_C.EVAL.SAVE_DIR = "./output"


def get_cfg_defaults():
  return _C.clone()
# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : Constant.py
# @Project: GOODKT
# @Comment :
Dpath = '../../Dataset'
datasets = {
    'assist2009' : 'assist2009',
    'assist2012' : 'assist2012',
    'assist2017' : 'assist2017',
    'assistednet': 'assistednet',
}

# question number of each dataset
numbers = {
    'assist2009' : 16891,
    'assist2012' : 37125,
    'assist2017' : 3162,
    'assistednet': 10795,
}

skill = {
    'assist2009' : 101,
    'assist2012' : 188,
    'assist2017' : 102,
    'assistednet': 1676,   # 188
}


DATASET = datasets['assist2017']
NUM_OF_QUESTIONS = numbers['assist2017']
H = '2017'

MAX_STEP = 50
BATCH_SIZE = 128
LR = 0.001
EPOCH = 20
EMB = 256
HIDDEN = 128  # sequence model's
kd_loss = 5.00E-06
LAYERS = 1

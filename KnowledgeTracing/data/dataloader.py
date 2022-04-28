# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : dataloader.py
# @Project: GOODKT
# @Comment :
import sys

sys.path.append('../')
import torch.utils.data as Data
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.preprocess import DataReader
from KnowledgeTracing.data.OneHot import OneHot


def getTrainLoader(train_data_path):
    handle = DataReader(train_data_path, C.MAX_STEP)
    trainques, trainans = handle.getTrainData()
    dtrain = OneHot(trainques, trainans)
    trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True, drop_last=True)
    return trainLoader


def getTestLoader(test_data_path):
    handle = DataReader(test_data_path, C.MAX_STEP)
    testques, testans = handle.getTestData()
    dtest = OneHot(testques, testans)
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False, drop_last=True)
    return testLoader


def getLoader(dataset):
    trainLoaders = []
    testLoaders = []
    if dataset == 'assist2009':
        trainLoader = getTrainLoader(C.Dpath + '/assist2009/assist2009_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2009/assist2009_pid_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'assist2017':
        trainLoader = getTrainLoader(C.Dpath + '/assist2017/assist2017_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2017/assist2017_pid_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'assistednet':
        trainLoader = getTrainLoader(C.Dpath + '/assistednet/assistednet_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assistednet/assistednet_pid_test.csv')
        testLoaders.append(testLoader)

    return trainLoaders[0], testLoaders[0]

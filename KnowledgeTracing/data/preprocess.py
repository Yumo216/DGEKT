# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : preprocess.py
# @Project: GOODKT
# @Comment :
import numpy as np
import itertools
import tqdm


class DataReader:
    def __init__(self, path, maxstep):
        self.path = path
        self.maxstep = maxstep


    def getTrainData(self):
        trainqus = np.array([])
        trainans = np.array([])

        with open(self.path, 'r', encoding='UTF-8-sig') as train:
            for len, ques, _, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='loading train data:    ',
                                               mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)

                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)

                trainqus = np.append(trainqus, ques).astype(int)
                trainans = np.append(trainans, ans).astype(int)
                trainqus = trainqus.reshape([-1, self.maxstep])
                trainans = trainans.reshape([-1, self.maxstep])
        return trainqus, trainans

    def getTestData(self):
        testqus = np.array([])
        testans = np.array([])
        with open(self.path, 'r', encoding='UTF-8-sig') as test:
            for len, ques, _, ans in tqdm.tqdm(itertools.zip_longest(*[test] * 4), desc='loading test data:    ',
                                               mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                testqus = np.append(testqus, ques).astype(int)
                testans = np.append(testans, ans).astype(int)
        return testqus.reshape([-1, self.maxstep]), testans.reshape([-1, self.maxstep])

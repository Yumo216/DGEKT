# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : OneHot.py
# @Project: GOODKT
# @Comment :
from torch.utils.data.dataset import Dataset
from KnowledgeTracing.Constant import Constants as C
import torch


class OneHot(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans
        self.numofques = C.NUM_OF_QUESTIONS

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        lab = self.onehot(questions, answers)
        return lab

    def onehot(self, questions, answers):
        label = torch.zeros(C.MAX_STEP, 2 * self.numofques).cuda()
        for i in range(C.MAX_STEP):
            if answers[i] > 0:
                label[i][questions[i]-1] = 1
            elif answers[i] == 0:
                label[i][self.numofques + questions[i]-1] = 1
        return label

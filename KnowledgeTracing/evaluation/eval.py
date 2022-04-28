# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : eval.py
# @Project: GOODKT
# @Comment :
import tqdm
import torch
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
from sklearn import metrics
import logging

logger = logging.getLogger('main.eval')


def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    logger.info('\nauc: ' + str(auc) + 'acc: ' + str(acc))
    print('auc: ' + str(auc) + ' acc: ' + str(acc))
    return auc, acc


class lossFunc(nn.Module):
    def __init__(self, hidden, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.q = C.NUM_OF_QUESTIONS
        self.hidden = hidden
        self.sig = nn.Sigmoid()
        self.max_step = max_step
        self.device = device
        self.mse = nn.MSELoss()

    def forward(self, logit_c, logit_t, logit_ensemble, batch):

        p_c = self.sig(logit_c)
        p_t = self.sig(logit_t)
        p_enm = self.sig(logit_ensemble)
        '''kd_loss'''
        T = 0.5
        p0_c = self.sig(logit_c/T)
        p0_t = self.sig(logit_t/T)
        p0_enm = self.sig(logit_ensemble/T)
        loss_kd = C.kd_loss * (torch.sum(torch.abs(p0_enm-p0_c)) + torch.sum(torch.abs(p0_enm-p0_t)))
        loss = torch.Tensor([0.0]).cuda()
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        for student in range(batch.shape[0]):
            delta = batch[student][:, :self.q] + batch[student][:, self.q:]

            a = (((batch[student][:, 0:self.q] -
                   batch[student][:, self.q:]).sum(1) + 1) //
                 2)[1:]  # [49]
            temp_c = p_c[student][:self.max_step - 1].mm(delta[1:].t())
            temp_t = p_t[student][:self.max_step - 1].mm(delta[1:].t())
            temp_enm = p_enm[student][:self.max_step - 1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                 dtype=torch.long, device=self.device)
            pc = temp_c.gather(0, index)[0]
            pt = temp_t.gather(0, index)[0]
            penm = temp_enm.gather(0, index)[0]
            for i in range(len(pc) - 1, -1, -1):
                if pc[i] > 0:
                    pc = pc[:i + 1]
                    pt = pt[:i + 1]
                    penm = penm[:i + 1]
                    a = a[:i + 1]
                    break
            loss = loss + self.crossEntropy(pt, a) + self.crossEntropy(pc, a)+ self.crossEntropy(penm, a)
            p_mean = (pt + pc + penm)/3.0
            prediction = torch.cat([prediction, p_mean])
            ground_truth = torch.cat([ground_truth, a])
        return loss, loss_kd,  prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func):
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        logit_c, logit_t, logit_ensemble = model(batch)
        loss, loss_kd,  prediction, ground_truth = loss_func(logit_c, logit_t, logit_ensemble, batch)
        total_loss = loss + loss_kd
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    return model, optimizer


def test_epoch(model, testLoader, loss_func, device):
    global loss_kd, total_loss
    ground_truth = torch.tensor([],device=device)
    prediction = torch.tensor([],device=device)
    loss = torch.tensor([],device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        logit_c, logit_t, logit_ensemble = model(batch)
        loss, loss_kd, p, a = loss_func(logit_c, logit_t, logit_ensemble, batch)
        total_loss = loss + loss_kd
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
    print('loss:', loss.item(), '   kd_loss:', loss_kd.item())
    print('Total loss:', total_loss.item())
    return performance(ground_truth, prediction)


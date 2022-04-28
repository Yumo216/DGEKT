# -*- coding: utf-8 -*-
# @Time : 2021/12/23 13:43
# @Author : Yumo
# @File : Model.py
# @Project: GOODKT
# @Comment :

from KnowledgeTracing.hgnn_models import HGNN
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
import torch
from KnowledgeTracing.DirectedGCN.GCN import GCN


class DKT(nn.Module):

    def __init__(self, hidden_dim, layer_dim, G, adj_in, adj_out):
        super(DKT, self).__init__()
        '''initial feature'''
        emb_dim = C.EMB
        emb = nn.Embedding(2 * C.NUM_OF_QUESTIONS, emb_dim)
        self.ques = emb(torch.LongTensor([i for i in range(2 * C.NUM_OF_QUESTIONS)])).cuda()
        '''generate two graphs'''
        self.G = G
        self.adj_out = adj_out
        self.adj_in = adj_in
        '''DGCN'''
        self.net1 = GCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2))
        self.net2 = GCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2))

        '''HGCN'''
        self.net = HGNN(in_ch=C.EMB,
                        n_hid=C.EMB,
                        n_class=C.EMB)
        '''GRU'''
        self.rnn1 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        self.rnn2 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        '''kd'''
        self.fc_c = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_t = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_ensemble = nn.Linear(2 * hidden_dim, C.NUM_OF_QUESTIONS)
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''SkillGraph: HGCN'''
        ques_h = self.net(self.ques, self.G)
        '''TransitionGraph: DGCN'''
        ques_out = self.net1(self.ques, self.adj_out)
        ques_in = self.net2(self.ques, self.adj_in)
        ques_d = torch.cat([ques_in, ques_out], -1)
        '''choose 50'''
        x_h = x.matmul(ques_h)
        x_d = x.matmul(ques_d)

        '''gru'''
        out_h, _ = self.rnn1(x_h)
        out_d, _ = self.rnn2(x_d)

        '''logits'''
        logit_c = self.fc_c(out_h)
        logit_t = self.fc_t(out_d)

        '''kd'''
        theta = self.sigmoid(self.w1(out_h) + self.w2(out_d))
        out_d = theta * out_d
        out_h = (1 - theta) * out_h
        emseble_logit = self.fc_ensemble(torch.cat([out_d, out_h], -1))

        return logit_c, logit_t, emseble_logit

# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : load_data.py
# @Project: GOODKT
# @Comment :
import numpy as np
import scipy.sparse as sp
from KnowledgeTracing.Constant import Constants as C
import tqdm
import itertools
import torch


def get_adj():
    q = C.NUM_OF_QUESTIONS
    resout = np.zeros((2 * q, 2 * q))
    path = '../../Dataset/' + C.DATASET + '/' + C.DATASET + '_pid_train.csv'

    with open(path, 'r', encoding='UTF-8-sig') as train:
        for len, ques, _, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='Generate adjacency matrix:    ',
                                           mininterval=2):
            len = int(len.strip().strip(','))
            ques = np.array(ques.strip().strip(',').split(',')).astype(int)
            ans = np.array(ans.strip().strip(',').split(',')).astype(int)
            if len > 1:
                for i in range(len):
                    if ans[i] == 0:
                        ques[i] += q

                for j in range(len - 1):
                    resout[ques[j] - 1][ques[j + 1] - 1] += 1
    resin = resout.T
    resout = normalize(resout + sp.eye(resout.shape[0]))
    resin = normalize(resin + sp.eye(resin.shape[0]))

    resout = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(resout))
    resin = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(resin))

    return resout, resin


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

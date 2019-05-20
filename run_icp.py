# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import time
import os
import multiprocessing
import sys
import numpy as np
from icp import ICPTrainer
import time
import tqdm
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--tgt_lang', type=str, default="es")
    parser.add_argument('--icp_init_epochs', type=int, default=100)
    parser.add_argument('--icp_train_epochs', type=int, default=50)
    parser.add_argument('--icp_ft_epochs', type=int, default=50)
    parser.add_argument('--n_pca', type=int, default=25)
    parser.add_argument('--n_icp_runs', type=int, default=100)
    parser.add_argument('--n_init_ex', type=int, default=5000)
    parser.add_argument('--n_ft_ex', type=int, default=7500)
    parser.add_argument('--n_eval_ex', type=int, default=20000)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--method', type=str, choices=["nn", "csls_knn_10"], default="csls_knn_10")
    parser.add_argument('--cp_dir', type=str, default="output")

    args = parser.parse_args()
    return args


def _run_icp(src_W, tgt_W, s0, i, n_pca, init_epochs):
    np.random.seed(s0 + i)
    icp = ICPTrainer(src_W.copy(), tgt_W.copy(), True, n_pca)
    t0 = time.time()
    indices_x, indices_y, rec, bb = icp.train_icp(init_epochs)
    dt = time.time() - t0
    print("%d: Rec %f BB %d Time: %f" % (i, rec, bb, dt))
    return indices_x, indices_y, rec, bb

def train_model(src_lang, tgt_lang, src_W, tgt_W, n_runs, n_pca, n_processes,
                init_epochs, train_epochs, ft_epochs, n_ft, min_rec=1e8, min_bb=None):
    data = np.zeros((params.n_runs, 2))

    best_idx_x = None
    best_idx_y = None

    s0 = np.random.randint(50000)
    results = []
    if params.n_processes == 1:
        for i in range(params.n_runs):
            results += [_run_icp(src_W, tgt_W, s0, i, n_pca, init_epochs)]
    else:
        pool = multiprocessing.Pool(processes=n_processes)
        for result in tqdm.tqdm(pool.imap_unordered(_run_icp, range(n_runs)), total=n_runs):
            results += [result]
        pool.close()

    for i, result in enumerate(results):
        indices_x, indices_y, rec, bb = result
        data[i, 0] = rec
        data[i, 1] = bb
        if rec < min_rec:
            best_idx_x = indices_x
            best_idx_y = indices_y
            min_rec = rec
            min_bb = bb

    idx = np.argmin(data[:, 0], 0)
    print("Init - Achieved: Rec %f BB %d" % (data[idx, 0], data[idx, 1]))
    icp_train = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
    _, _, rec, bb = icp_train.train_icp(train_epochs, True, best_idx_x, best_idx_y)
    print("Training - Achieved: Rec %f BB %d" % (rec, bb))
    src_W = np.load("data/%s_%d.npy" % (src_lang, n_ft)).T
    tgt_W = np.load("data/%s_%d.npy" % (tgt_lang, n_ft)).T
    icp_ft = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
    icp_ft.icp.TX = icp_train.icp.TX
    icp_ft.icp.TY = icp_train.icp.TY
    _, _, rec, bb = icp_ft.train_icp(ft_epochs, do_reciprocal=True)
    print("Reciprocal Pairs - Achieved: Rec %f BB %d" % (rec, bb))
    TX = icp_ft.icp.TX
    TY = icp_ft.icp.TY

    return TX, TY


if __name__ == "__main__":
    params = parse_args()

    src_W = np.load("data/%s_%d.npy" % (params.src_lang, params.n_init_ex)).T
    tgt_W = np.load("data/%s_%d.npy" % (params.tgt_lang, params.n_init_ex)).T

    TX, TY = train_model(params.src_lang, params.tgt_lang, src_W, tgt_W, params.n_icp_runs, params.n_pca,
                         params.n_processes, params.icp_init_epochs, params.icp_train_epochs, params.icp_ft_epochs,
                         params.n_ft_ex)

    if not os.path.exists(params.cp_dir):
        os.mkdir(params.cp_dir)

    np.save("%s/%s_%s_T" % (params.cp_dir, params.src_lang, params.tgt_lang), TX)
    np.save("%s/%s_%s_T" % (params.cp_dir, params.tgt_lang, params.src_lang), TY)






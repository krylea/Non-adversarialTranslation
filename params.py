# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

'''
src_lang = "en"
tgt_lang = "es"
icp_init_epochs = 100
icp_train_epochs = 50
icp_ft_epochs = 50
n_pca = 25
n_icp_runs = 100
n_init_ex = 5000
n_ft_ex = 7500
n_eval_ex = 200000
n_processes = 1
method = 'csls_knn_10' # nn|csls_knn_10
cp_dir = "output"
'''


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
    parser.add_argument('--method', type=str, choices=["nn", "csls_knn_10"], default="csls_knn_10")
    parser.add_argument('--cp_dir', type=str, default="output")

    args = parser.parse_args()
    return args
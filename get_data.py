# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import utils
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--tgt_lang', type=str, default="es")
    args = parser.parse_args()
    return args

params = parse_args()

src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_init_ex, False)
np.save('data/%s_%d' % (params.src_lang, params.n_init_ex), src_embeddings)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_init_ex, False)
np.save('data/%s_%d' % (params.tgt_lang, params.n_init_ex), tgt_embeddings)

src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_ft_ex, False)
np.save('data/%s_%d' % (params.src_lang, params.n_ft_ex), src_embeddings)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_ft_ex, False)
np.save('data/%s_%d' % (params.tgt_lang, params.n_ft_ex), tgt_embeddings)

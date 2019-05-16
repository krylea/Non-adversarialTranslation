# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import utils
import argparse
import os
import wget

vec_dir = "data"

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--tgt_lang', type=str, default="es")
    parser.add_argument('--n_init_ex', type=int, default=5000)
    parser.add_argument('--n_ft_ex', type=int, default=7500)
    args = parser.parse_args()
    return args

params = parse_args()

src_path = os.path.join(vec_dir, 'wiki.{}.vec'.format(params.src_lang))
tgt_path = os.path.join(vec_dir, 'wiki.{}.vec'.format(params.tgt_lang))
for l, p in [(params.src_lang, src_path), (params.tgt_lang, tgt_path)]:
    # Download embeddings if not present
    if not os.path.exists(p):
        vec_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'.format(l)
        try:
            _ = wget.download(vec_url, out=vec_dir)
        except:
            print('Download of {} vectors from {} failed.'.format(l, vec_url))

src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_init_ex, False)
np.save('data/%s_%d' % (params.src_lang, params.n_init_ex), src_embeddings)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_init_ex, False)
np.save('data/%s_%d' % (params.tgt_lang, params.n_init_ex), tgt_embeddings)

src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_ft_ex, False)
np.save('data/%s_%d' % (params.src_lang, params.n_ft_ex), src_embeddings)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_ft_ex, False)
np.save('data/%s_%d' % (params.tgt_lang, params.n_ft_ex), tgt_embeddings)

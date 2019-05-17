import torch
import numpy as np

import utils


def procrustes(src_emb, tgt_emb, mapping, pairs):
    """
    Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """
    A = src_emb[pairs[:, 0]]
    B = tgt_emb[pairs[:, 1]]
    W = mapping
    M = B.transpose(0, 1).mm(A)
    U, S, V = torch.svd(torch.Tensor(M))
    #scipy.linalg.svd(M, full_matrices=True)

    return (U.mm(V.t()).type_as(W)).numpy()

def load_models(cp_dir, src_lang, tgt_lang):
    T_s2t = np.load("%s/%s_%s_T.npy" % (cp_dir, src_lang, tgt_lang))
    T_t2s = np.load("%s/%s_%s_T.npy" % (cp_dir, tgt_lang, src_lang))
    return T_s2t, T_t2s

def load_data(src_lang, tgt_lang, n_eval_ex=20000):
    src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % src_lang,
                                                                         n_eval_ex, False)
    tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % tgt_lang,
                                                                         n_eval_ex, False)

    s2t_dict = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (src_lang, tgt_lang), src_word2id,
                                       tgt_word2id)
    t2s_dict = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (tgt_lang, src_lang), tgt_word2id,
                                     src_word2id)
    return (src_id2word, src_word2id, src_embeddings), (tgt_id2word, tgt_word2id, tgt_embeddings), s2t_dict, t2s_dict

def evaluate(T_s2t, T_t2s, src_lang, tgt_lang, src_data, tgt_data, s2t_dict, t2s_dict, method="csls_knn_10"):
    src_id2word, src_word2id, src_embeddings = src_data
    tgt_id2word, tgt_word2id, tgt_embeddings = tgt_data

    TranslatedX = src_embeddings.dot(np.transpose(T_s2t))
    TranslatedY = tgt_embeddings.dot(np.transpose(T_t2s))

    s2t_nn = utils.get_word_translation_accuracy(src_lang, src_word2id, TranslatedX,
                                                   tgt_lang, tgt_word2id, tgt_embeddings,
                                                   "nn", s2t_dict)
    s2t_csls = utils.get_word_translation_accuracy(src_lang, src_word2id, TranslatedX,
                                                   tgt_lang, tgt_word2id, tgt_embeddings,
                                                   "csls_knn_10", s2t_dict)

    t2s_nn = utils.get_word_translation_accuracy(tgt_lang, tgt_word2id, TranslatedY,
                                                 src_lang, src_word2id, src_embeddings,
                                                 "nn", t2s_dict)
    t2s_csls = utils.get_word_translation_accuracy(tgt_lang, tgt_word2id, TranslatedY,
                                                   src_lang, src_word2id, src_embeddings,
                                                   "csls_knn_10", t2s_dict)

    return s2t_nn, s2t_csls, t2s_nn, t2s_csls
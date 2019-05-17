import torch
import numpy as np


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

def load_model(cp_dir, src_lang, tgt_lang):
    T = np.load("%s/%s_%s_T.npy" % (cp_dir, src_lang, tgt_lang))
    return T


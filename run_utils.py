import torch
import numpy as np

import utils

USE_CUDA = torch.cuda.is_available()

def cosine_knn_muse(query, vectors, k=5, vocab=None):
    if USE_CUDA:
        query = query.cuda()
        vectors = vectors.cuda()

    #scores = query.matmul(vectors)
    v_norm = torch.norm(vectors, dim=1)
    c_norm = torch.norm(query).expand_as(v_norm)
    dotprod = query.matmul(vectors.transpose(0, 1))
    dists = -1 * dotprod / (c_norm * v_norm)

    return (-1 * dists).topk(k)

def _get_candidates(vectors1, vectors2, dist=cosine_knn_muse, bs=128, max_rank=15000, min_size=0, max_size=0, threshold=0):
    all_scores = []
    all_targets = []

    n = vectors1.size(0)
    for i in range(0, n, bs):
        scores, targets = dist(vectors1[i:min(n, i+bs)], vectors2, k=2)
        all_scores.append(scores.cpu())
        all_targets.append(targets.cpu())

    all_scores = torch.cat(all_scores,0)
    all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    assert all_scores.size() == all_pairs.size() == (n, 2)

    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if max_rank > 0:
        selected = all_pairs.max(1)[0] <= max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if max_size > 0:
        all_scores = all_scores[:max_size]
        all_pairs = all_pairs[:max_size]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if min_size > 0:
        diff[:min_size] = 1e9

    # confidence threshold
    if threshold > 0:
        mask = diff > threshold
        print("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs.cuda() if USE_CUDA else all_pairs

def _build_dictionary(src_emb, tgt_emb, mode, s2t_candidates=None, t2s_candidates=None, **kwargs):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    #print("Building the train dictionary ...")
    s2t = 'S2T' in mode
    t2s = 'T2S' in mode
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = _get_candidates(src_emb, tgt_emb, **kwargs)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = _get_candidates(tgt_emb, src_emb, **kwargs)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if mode == 'S2T':
        dico = s2t_candidates
    elif mode == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])
        if mode == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert mode == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                print("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    #print('New train dictionary of %i pairs.' % dico.size(0))
    return dico


def build_dictionary(src_emb, tgt_emb, mapping, mode, bs=64, max_rank=15000, threshold=0):
    """
    Build a dictionary from aligned embeddings.
    """
    dico_kwargs = {'max_rank': max_rank, 'threshold': threshold}


    if mode == 'S2T':
        s2t_vecs = src_emb.mm(mapping.t())
        tgt_vecs = tgt_emb.data
        s2t_vecs = s2t_vecs / s2t_vecs.norm(2, 1, keepdim=True).expand_as(s2t_vecs)
        tgt_vecs = tgt_vecs / tgt_vecs.norm(2, 1, keepdim=True).expand_as(tgt_vecs)
        s2t_dico = _build_dictionary(s2t_vecs, tgt_vecs, mode=mode, bs=bs, **dico_kwargs)
        return s2t_dico
    elif mode == 'T2S':
        t2s_vecs = tgt_emb.mm(mapping.t())
        src_vecs = src_emb.data
        t2s_vecs = t2s_vecs / t2s_vecs.norm(2, 1, keepdim=True).expand_as(t2s_vecs)
        src_vecs = src_vecs / src_vecs.norm(2, 1, keepdim=True).expand_as(src_vecs)
        t2s_dico = _build_dictionary(t2s_vecs, src_vecs,  mode='S2T', bs=bs, **dico_kwargs)
        return t2s_dico
    elif mode == 'joint':
        s2t_mapping, t2s_mapping = mapping

        s2t_dico = build_dictionary(src_emb, tgt_emb, s2t_mapping, 'S2T', **dico_kwargs)
        t2s_dico = build_dictionary(src_emb, tgt_emb, t2s_mapping, 'T2S', **dico_kwargs)

        s2t_set = set([(a, b) for a, b in s2t_dico.cpu().numpy()])
        t2s_set = set([(b, a) for a, b in t2s_dico.cpu().numpy()])

        joint_set = s2t_set & t2s_set
        joint_dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in joint_set]))
        joint_dico = joint_dico.cuda() if USE_CUDA else joint_dico
        joint_rev = joint_dico[:,[1,0]]
        return s2t_dico, t2s_dico, joint_dico, joint_rev


def _procrustes(src_emb, tgt_emb, mapping, pairs):
    """
    Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """
    A = src_emb[pairs[:, 0]]
    B = tgt_emb[pairs[:, 1]]
    W = mapping
    M = B.transpose(0, 1).mm(A)
    U, S, V = torch.svd(M)
    #scipy.linalg.svd(M, full_matrices=True)

    return (U.mm(V.t()).type_as(W))

def procrustes(src_emb, tgt_emb, mapping, iters, dico=None):
    dico = build_dictionary(src_emb, tgt_emb, mapping, "S2T") if dico is None else dico
    for i in range(iters):
        mapping = _procrustes(src_emb, tgt_emb, mapping, dico)
        dico = build_dictionary(src_emb, tgt_emb, mapping, "S2T")
    return mapping, dico

def joint_refinement(src_emb, tgt_emb, s2t_mapping, t2s_mapping, s2t_pairs, t2s_pairs, iters):
    src_emb, tgt_emb, s2t_mapping, t2s_mapping = [torch.Tensor(x).cuda() if USE_CUDA else torch.Tensor(x)
                                                  for x in (src_emb, tgt_emb, s2t_mapping, t2s_mapping)]

    s2t, t2s, joint_s2t, joint_t2s = build_dictionary(src_emb, tgt_emb, (s2t_mapping, t2s_mapping), "joint")

    proc_s2t_map, proc_s2t_dico = procrustes(src_emb, tgt_emb, s2t_mapping, iters, dico=s2t)
    proc_t2s_map, proc_t2s_dico = procrustes(tgt_emb, src_emb, t2s_mapping, iters, dico=t2s)
    proc_out = evaluate(proc_s2t_map, proc_t2s_map, src_emb, tgt_emb, s2t_pairs,
                                                  t2s_pairs)
    del proc_s2t_map,proc_s2t_dico,proc_t2s_map,proc_t2s_dico

    joint_proc_s2t_map, joint_proc_s2t_dico = procrustes(src_emb, tgt_emb, s2t_mapping, iters, dico=joint_s2t)
    joint_proc_t2s_map, joint_proc_t2s_dico = procrustes(tgt_emb, src_emb, t2s_mapping, iters, dico=joint_t2s)
    joint_out = evaluate(joint_proc_s2t_map, joint_proc_t2s_map, src_emb, tgt_emb,
                                                                          s2t_pairs, t2s_pairs)
    del joint_proc_s2t_map, joint_proc_s2t_dico, joint_proc_t2s_map, joint_proc_t2s_dico

    return {"proc":proc_out,"joint": joint_out}



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
    return src_embeddings, tgt_embeddings, s2t_dict, t2s_dict

def evaluate(T_s2t, T_t2s, src_emb, tgt_emb, s2t_dict, t2s_dict):
    src_emb, tgt_emb, s2t_dict, t2s_dict, T_s2t, T_t2s = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                                                          for x in (src_emb, tgt_emb, s2t_dict, t2s_dict, T_s2t, T_t2s)]
    TranslatedX = src_emb.dot(np.transpose(T_s2t))
    TranslatedY = tgt_emb.dot(np.transpose(T_t2s))

    s2t_nn = utils.get_word_translation_accuracy(TranslatedX, tgt_emb, "nn", s2t_dict)
    s2t_csls = utils.get_word_translation_accuracy(TranslatedX, tgt_emb, "csls_knn_10", s2t_dict)

    t2s_nn = utils.get_word_translation_accuracy(TranslatedY, src_emb, "nn", t2s_dict)
    t2s_csls = utils.get_word_translation_accuracy(TranslatedY, src_emb, "csls_knn_10", t2s_dict)

    return {"s2t_nn": s2t_nn, "s2t_csls": s2t_csls, "t2s_nn": t2s_nn, "t2s_csls": t2s_csls}
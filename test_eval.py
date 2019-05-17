from run_utils import *

src_lang = "en"
tgt_lang = "fr"
out_dir = "output"
N = 200000
iters=1

T_s2t, T_t2s = load_models(out_dir, src_lang, tgt_lang)
src_emb, tgt_emb, s2t_dict, t2s_dict = load_data(src_lang, tgt_lang, N)
s2t_nn, s2t_csls, t2s_nn, t2s_csls = evaluate(T_s2t, T_t2s, src_emb, tgt_emb, s2t_dict, t2s_dict)
out = joint_refinement(src_emb, tgt_emb, T_s2t, T_t2s, iters)
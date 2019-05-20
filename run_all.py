from run_icp import train_model
from run_utils import *

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--total_runs', type=int, default=25)
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--tgt_lang', type=str, default="es")
    parser.add_argument('--icp_init_epochs', type=int, default=100)
    parser.add_argument('--icp_train_epochs', type=int, default=50)
    parser.add_argument('--icp_ft_epochs', type=int, default=50)
    parser.add_argument('--n_pca', type=int, default=25)
    parser.add_argument('--n_icp_runs', type=int, default=100)
    parser.add_argument('--n_init_ex', type=int, default=5000)
    parser.add_argument('--n_ft_ex', type=int, default=7500)
    parser.add_argument('--vocab_size', type=int, default=200000)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--proc_iters', type=int, default=5)
    parser.add_argument('--method', type=str, choices=["nn", "csls_knn_10"], default="csls_knn_10")
    parser.add_argument('--cp_dir', type=str, default="output")

    args = parser.parse_args()
    return args


def save_out(out_dir, scores):
    with open(os.path.join(out_dir, "results.txt"),'a') as outfile:
        for k,v in scores.items():
            outfile.write(k+":\n")
            for k2,v2 in v.items():
                outfile.write("\t%s:%f"%(k2,v2))


if __name__ == "__main__":
    params = parse_args()

    src_emb, tgt_emb, s2t_dict, t2s_dict = load_data(params.src_lang, params.tgt_lang, params.vocab_size)

    for i in range(params.total_runs):
        TX, TY = train_model(params.src_lang, params.tgt_lang, src_emb, tgt_emb, params.n_icp_runs, params.n_pca,
                             params.n_processes, params.icp_init_epochs, params.icp_train_epochs, params.icp_ft_epochs,
                             params.n_ft_ex)

        out_dir=os.path.join(params.cp_dir, i)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        np.save("%s/%s_%s_T" % (out_dir, params.src_lang, params.tgt_lang), TX)
        np.save("%s/%s_%s_T" % (out_dir, params.tgt_lang, params.src_lang), TY)

        base_scores = evaluate(TX, TY, src_emb, tgt_emb, s2t_dict, t2s_dict)
        proc_scores = joint_refinement(src_emb, tgt_emb, TX, TY, s2t_dict, t2s_dict, params.proc_iters)
        proc_scores['base'] = base_scores

        save_out(out_dir, proc_scores)
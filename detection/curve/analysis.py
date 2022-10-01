import os
import sys
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append("../..")

import argparse
parser = argparse.ArgumentParser(description="Poisoning Benchmark")
parser.add_argument('--num', type=int, default=5)
parser.add_argument('--N_model', type=int, default=25)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--benign_path', required=True)
parser.add_argument('--trojan_path', required=True)
args = parser.parse_args()

from tools import *

type = 'clean'

th = 3

shuffle_list = list(range(1, args.num))
num_class = args.num_class


def main(save_path, seed, name, draw=False):

    distance_all = torch.empty([num_class, len(shuffle_list)])
    for s_idx, shuffle_rate in enumerate(shuffle_list):
        root = os.path.join(save_path, '%s_%d_features'%('last', shuffle_rate))
        distance_all[:, s_idx] = torch.load(os.path.join(root, 'seed_%d'%(seed))).cpu()

    re = mad(distance_all, seed, name, draw=draw)

    return np.max(re)


if __name__ == '__main__':

    N_model = args.N_model

    path_list = [args.benign_path, args.trojan_path]
    scores = np.empty([len(path_list), N_model])
    for idx, name in enumerate(path_list):
        path = '%s/save/std'%name
        for seed in range(N_model):
            score = main(path, seed, 'benign' if idx==0 else 'backdoor', False)
            scores[idx, seed] = score


    for idx in range(1, len(path_list)):
        scores_temp = np.concatenate([scores[0], scores[idx]], axis=0)
        label = np.concatenate([np.zeros([N_model]), np.ones([N_model])], axis=0)
        auroc = roc_auc_score(label, scores_temp)

        fpr,tpr,threshold = roc_curve(label, scores_temp) 

        print(auroc)
    
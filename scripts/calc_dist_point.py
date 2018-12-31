import os
from os.path import join, abspath
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import glob
import argparse
import pickle as pkl
from functools import partial
from itertools import chain

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import trange

from audiodistances import dtw_fast, simple_fast, mckl
from audiodistances.config import Config as cfg
from audiodistances.utils import parmap


simple_fast_ = partial(simple_fast, sub_seq_len=50)
mckl_ = partial(mckl, n_components=cfg.GMM_COMPONENTS, r=cfg.MC_SAMPLES)

DIST_TYPES = {
    'MCKL': lambda a, b: mckl_(model_a=a, model_b=b),
    'SIMPLE': lambda a, b: np.median(simple_fast_(a, b)[0]),
    'DTW': dtw_fast
}
LOAD_FUNC = {
    'MCKL': lambda fn: pkl.load(open(fn, 'rb')),
    'SIMPLE': np.load,
    'DTW': np.load
}


def _calc_dist(ij, all_fns, dist_type):
    """Calculate pair-wise distance from two music files

    Args:
        ij (tuple of int): index of target items
        all_fns (list of str): filename list
        dist_type (str): type of distance {'MCKL', 'SIMPLE', 'DTW'}
    
    Returns:
        float: distance
    """
    # parse data
    i, j = ij

    # retrieve distance function
    dist_fnc = DIST_TYPES[dist_type]

    # load audio features (n_steps, n_dim)
    X1, X2 = LOAD_FUNC[dist_type](all_fns[i]), LOAD_FUNC[dist_type](all_fns[j])

    # calc dist
    d_ij = dist_fnc(X1, X2)
    if dist_type == 'SIMPLE':  # it's asymmetric
        d_ji = dist_fnc(X2, X1)
    else:
        d_ji = d_ij

    return (i, j ,(d_ij, d_ji))


def process(music_features, target_index, sample_rate=1,
            dist_type='SIMPLE', n_jobs=1):
    """Calculate pairwise distances between music files
    """
    # 1) prepare targets
    candidate_pairs = ((target_index, other_index)
                       for other_index in range(len(music_features)))

    # 2) calculate distances
    print('Main Process!')
    res = parmap(
        partial(_calc_dist, all_fns=music_features, dist_type=dist_type),
        candidate_pairs, total= len(music_features),
        verbose=True, n_workers=n_jobs
    )

    # 3) dump result
    output = []
    for (i, j, (d_ij, d_ji)) in res:
        output.append((i, j, d_ij))
        output.append((j, i, d_ji))
    return output


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_features",
                        help='text file contains all the file name of \
                              music features or gmms (for MCKL)')
    parser.add_argument("target_index", type=int,
                        help="index of target original file to calc \
                              slice of distances")
    parser.add_argument("num_originals", type=int,
                        help='number of original samples from top of \
                              the given feature filename list')
    parser.add_argument("distance", type=str, default='MCKL',
                        choices=set(DIST_TYPES.keys()),
                        help='type indication for distance \
                            {MCKL, SIMPLE, DTW}')
    parser.add_argument("out_fn", help='filename of dump output file')
    parser.add_argument("--n-jobs", type=int, default=1, help='number of parallel jobs')
    args = parser.parse_args()

    # sanity check
    if args.target_index >= args.num_originals:
        raise ValueError('[ERROR] target item should be one of the \
                          original items! (target_index >= num_originals)')

    # load the file list
    with open(args.music_features) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    result = process(fns, args.target_index, args.num_originals,
                     dist_type=args.distance, n_jobs=args.n_jobs)

    # write the result file
    with open(args.out_fn, 'wb') as f:
        pkl.dump(result, f)

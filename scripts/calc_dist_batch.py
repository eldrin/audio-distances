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

import numpy as np
from sklearn.mixture import GaussianMixture

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


def _calc_dist(fns, dist_type):
    """Calculate pair-wise distance from two music files

    Args:
        fn1 (str): music feature file name 1
        fn2 (str): music feature file name 2
        dist_type (str): type of distance {'MCKL', 'SIMPLE', 'DTW'}
    
    Returns:
        float: distance
    """
    # parse data
    fn1, fn2 = fns

    # retrieve distance function
    dist_fnc = DIST_TYPES[dist_type]

    # load audio features (n_steps, n_dim)
    X1, X2 = LOAD_FUNC[dist_type](fn1), LOAD_FUNC[dist_type](fn2)

    return dist_fnc(X1, X2)


def process(music_features, num_originals, sample_rate=1,
            dist_type='SIMPLE', n_jobs=1):
    """Calculate pairwise distances between music files
    """
    fns = music_features

    # 1) prepare the process
    og = fns[:num_originals]

    # 1.1) sorting out the candidates
    candidate_pairs = []
    for fn1 in og:
        for fn2 in fns:
            # sampling
            if sample_rate < 1 and ~np.random.binomial(1, p=sample_rate):
                continue
            candidate_pairs.append((fn1, fn2))
            candidate_pairs.append((fn2, fn1))
    fn_hash = {fn:i for i, fn in enumerate(fns)}

    # 2) calculate distances
    res = parmap(
        partial(_calc_dist, dist_type=dist_type),
        candidate_pairs, verbose=True, n_workers=n_jobs
    )

    # 3) dump result
    output = []
    for (fn1, fn2), d in zip(candidate_pairs, res):
        output.append((fn_hash[fn1], fn_hash[fn2], d))

    return output


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_features",
                        help='text file contains all the file name of \
                              music features or gmms (for MCKL)')
    parser.add_argument("num_originals", type=int,
                        help='number of original samples from top of \
                              the given feature filename list')
    parser.add_argument("distance", type=str, default='MCKL',
                        choices=set(DIST_TYPES.keys()),
                        help='type indication for distance \
                            {MCKL, SIMPLE, DTW}')
    parser.add_argument("out_fn", help='filename of dump output file')
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    args = parser.parse_args()

    # load the file list
    with open(args.music_features) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    result = process(fns, args.num_originals,
                     dist_type=args.distance, n_jobs=args.n_jobs)

    # write the result file
    with open(args.out_fn, 'wb') as f:
        pkl.dump(result, f)

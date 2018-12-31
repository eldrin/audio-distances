import os
from os.path import join, abspath, basename, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from functools import partial
import pickle as pkl
import numpy as np
from sklearn.mixture import GaussianMixture

from audiodistances.utils import parmap
from audiodistances.config import Config as cfg


def _fit_gmm(fn, out_root):
    """Fit a GMM and cache to the file

    Args:
        fn (str): music feature file name
        out_fn (str): file name to dump the gmm
    """
    out_fn = join(out_root, splitext(basename(fn))[0] + '_gmm.pkl')
    X = np.load(fn)  # load the feature file (n_steps, n_dim)
    gmm = GaussianMixture(n_components=cfg.GMM_COMPONENTS).fit(X)
    with open(out_fn, 'wb') as f:
        pkl.dump(gmm, f)


def fit_gmms(fns, out_root, n_jobs=1):
    """Extract MFCCs

    Args:
        fns (str): file name of the music
        out_root (str): path to dump gmms
        n_jobs (int): number of parallel jobs
    """
    parmap(
        partial(_fit_gmm, out_root=out_root),
        fns, n_workers=n_jobs, verbose=True
    )


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_features",
                        help='text file contains all the file names of music features')
    parser.add_argument("out_path", help='path to dump output files')
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    args = parser.parse_args() 

    # load the file list
    with open(args.music_features) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    # process!
    fit_gmms(fns, args.out_path, n_jobs=args.n_jobs)

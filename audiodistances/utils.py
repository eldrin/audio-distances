from os.path import join, dirname, basename
import shutil
import pickle as pkl
from multiprocessing import Pool

from tqdm import tqdm


def parmap(func, iterable, n_workers=2, verbose=False):
    """ Simple Implementation for Parallel Map """
    
    if n_workers == 1:
        if verbose:
            iterable = tqdm(iterable, total=len(iterable), ncols=80)
        return map(func, iterable)
    else:
        with Pool(processes=n_workers) as p:
            if verbose:
                with tqdm(total=len(iterable), ncols=80) as pbar:
                    output = []
                    for o in p.imap_unordered(func, iterable):
                        output.append(o)
                        pbar.update()
                return output
            else:
                return p.imap_unordered(func, iterable)
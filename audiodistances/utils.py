from os.path import join, dirname, basename
import shutil
import pickle as pkl
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm


def parmap(func, iterable, total=None, n_workers=2, verbose=False):
    """ Simple Implementation for Parallel Map """
    if total:
        tqdm_ = partial(tqdm, total=total, ncols=80)
    else:
        try:
            tqdm_ = partial(tqdm, total=len(iterable), ncols=80)
        except Exception as e:
            print(e)
        finally:
            tqdm_ = partial(tqdm, ncols=80)
    
    if n_workers == 1:
        if verbose:
            iterable = tqdm_(iterable)
        return map(func, iterable)
    else:
        with Pool(processes=n_workers) as p:
            if verbose:
                with tqdm_() as pbar:
                    output = []
                    for o in p.imap_unordered(func, iterable):
                        output.append(o)
                        pbar.update()
                return output
            else:
                return p.imap_unordered(func, iterable)

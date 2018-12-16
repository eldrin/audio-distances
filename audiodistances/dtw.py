import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def dtw_fast(A, B):
    """Fast-DTW to calculate distance between two sequences A, B

    Args:
        A (numpy.ndarray): input sequence having size of (n_steps, n_feature)
        B (numpy.ndarray): another sequence that has same feature dimension as A
    
    Returns:
        float: symmetric distance between two sequence A and B
    """
    return fastdtw(A, B, dist=euclidean)[0]


def test_dtw():
    """Testing DTW
    """
    A = np.random.randn(10, 2)
    B = np.random.randn(10, 2)
    print(dtw_fast(A, B))


if __name__ == "__main__":
    test_dtw()
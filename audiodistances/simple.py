import numpy as np


def simple_fast(A, B, sub_seq_len):
    """Compute the similarity join of a given time series A with another time series B

    NOTE: this version is directly ported from original implementation in MATLAB

    Args:
        A (numpy.ndarray): input sequence having size of (n_steps, n_feature)
        B (numpy.ndarray): another sequence that has same feature dimension as A
    
    Returns:
        numpy.ndarray: matrix profile of the join (n_steps,)
        numpy.ndarray: matrix profile of the join (n_steps,)

    .. _SiMPle-Fast
       https://sites.google.com/view/simple-fast
    """
    # we want the features in the columns
    if A.shape[1] > A.shape[0]:
        A, B = A.T, B.T
    
    # initialization
    matrix_profile_len = A.shape[0] - sub_seq_len + 1
    matrix_profile = np.zeros((matrix_profile_len,))
    mp_index = np.zeros((matrix_profile_len,))
    n_d = A.shape[1]

    # for the first dot-product of each sub-seq
    X, n, sumx2 = fast_find_nn_pre(A, sub_seq_len)
    sub_seq = B[:sub_seq_len]
    _, firstz, _, _ = fast_find_nn(X, sub_seq, n, sub_seq_len, n_d, sumx2)

    # compute necessary values
    X, n, sumx2 = fast_find_nn_pre(B, sub_seq_len)

    # compute first distance profile
    sub_seq = A[:sub_seq_len]
    dist_profile, currz, drop_val, sumy2 = (
        fast_find_nn(X, sub_seq, n, sub_seq_len, n_d, sumx2)
    )
    matrix_profile[0] = dist_profile.min()
    mp_index[0] = np.argmin(dist_profile)
    nz = currz.shape[0]

    # compute the remainder of the matrix profile
    for i in range(1, matrix_profile_len):
        sub_seq = A[i:i+sub_seq_len]
        sumy2 = sumy2 - drop_val**2 + sub_seq[-1]**2

        currz[1:nz] = (
            currz[:nz-1] +
            sub_seq[-1] * B[sub_seq_len:sub_seq_len + nz -1] -
            drop_val * B[:nz-1]
        )
        currz[0] = firstz[i]
        drop_val = sub_seq[0]
        dist_profile = np.sum(sumx2 - 2 * currz + sumy2, axis=1)
        
        matrix_profile[i] = dist_profile.min()
        mp_index[i] = np.argmin(dist_profile)

    return matrix_profile, mp_index


def fast_find_nn_pre(x, m):
    """Pre-processing step to find first NN

    This function is modified from the code provided in FasterSimilaritySearch

    Args:
        x (numpy.ndarray): input sequence
        m (int): sub-sequence length
    
    Returns:
        numpy.ndarray: FFT of given sequence x
        int: length of the given sequence x
        numpy.ndarray: sum of data x

    .. _FasterSimilaritySearch:
       http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    """
    n, n_d = x.shape
    # padding
    x = np.r_[x, np.zeros(x.shape)]

    X = np.fft.fft(x, axis=0)
    cum_sumx2 = np.cumsum(x**2, axis=0)
    sumx2 = cum_sumx2[m-1:n] - np.r_[np.zeros((1, n_d)), cum_sumx2[:n - m]]
    return X, n, sumx2


def fast_find_nn(X, y, n, m, n_d, sumx2):
    """Finding Nearest Neighbor using FFT convolution

    This function is modified from the code provided in FasterSimilaritySearch

    Args:
        X (numpy.ndarray): database sequence
        y (numpy.ndarray): query sequence
        n (int): length of dataset sequence
        m (int): query sequence length
        n_d (int): dimensionality of vector of each time-step
        sumx2 (numpy.ndarray): cached sum of x^2
    
    Returns:
        numpy.ndarray: distance over time steps (n,)
        numpy.ndarray: similarity after search result
        numpy.ndarray: dropped value
        numpy.ndarray: sum of sequence y^2

    .. _FasterSimilaritySearch:
       http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    """
    drop_val = y[0].ravel()
    # x is the data, y is the query
    y = y[::-1]
    # padding
    y = np.r_[y, np.zeros((2*n - m, n_d))]

    # the main trick of getting dot products in O(n log n) time
    Y = np.fft.fft(y, axis=0)
    Z = X * Y
    z = np.fft.ifft(Z, axis=0)

    # compute y stats -- O(n)
    sumy2 = np.sum(y**2, axis=0)

    # computing the distances -- O(n) time
    z = np.real(z[m-1:n])
    # keyboard
    dist = np.sum(sumx2 - 2 * z + sumy2, axis=1)

    return dist, z, drop_val, sumy2


def test_simple_fast():
    """Testing the functionality of SiMPle-Fast
    """
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([[5, 6], [7, 8], [9, 10]])
    res = simple_fast(A, B, 1)
    print(res)


if __name__ == "__main__":
    test_simple_fast()
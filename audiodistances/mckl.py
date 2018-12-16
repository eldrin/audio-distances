import numpy as np
from sklearn.mixture import GaussianMixture


def mckl(A=None, B=None, model_a=None, model_b=None, n_components=5, r=256):
    """Monte-Carlo sampling based KL-divergence likelihood ratio

    Args:
        A, B (numpy.ndarray): input matrices to compare (n_timestep, n_dim)
        model_a, model_b (GaussianMixture): pre-trained GMM model for each sequence
        n_components (int) : number of gaussian mixture for GMM
        r (int): number of samples to draw for approximation
    
    Returns:
        float: symmetric distance between two sequence A and B
    """
    if A is None or B is None:
        if model_a is None or model_b is None:
            raise ValueError('[ERROR] data (A, B) or model \
                            (model_a, model_b) pair should be given!')
    else:
        if not (A is not None and B is not None):
            raise ValueError('[ERROR] data (A, B) or model \
                            (model_a, model_b) pair should be given!')

    # train GMMs
    if model_a is None:
        model_a = GaussianMixture(n_components).fit(A)
    elif not isinstance(model_a, GaussianMixture):
        model_a = GaussianMixture(n_components).fit(A)

    if model_b is None:
        model_b = GaussianMixture(n_components).fit(B)
    elif not isinstance(model_b, GaussianMixture):
        model_b = GaussianMixture(n_components).fit(B)

    # draw samples
    x_a = model_a.sample(r)[0]
    x_b = model_b.sample(r)[0]

    # calc distance
    Dab = model_a.score(x_a) - model_b.score(x_a)
    Dba = model_b.score(x_b) - model_a.score(x_b)
    return Dab + Dba


def test_mckl():
    """Testing the MCKL distance
    """
    # directly using 
    A = np.random.randn(10, 2)
    B = np.random.randn(10, 2)
    d1 = mckl(A, B, n_components=3, r=5)

    # using pre-trained GMMs
    gmm_a = GaussianMixture(3).fit(A)
    gmm_b = GaussianMixture(3).fit(B)
    d2 = mckl(model_a=gmm_a, model_b=gmm_b, n_components=3, r=5)

    print(d1, d2)

    
if __name__ == "__main__":
    test_mckl()
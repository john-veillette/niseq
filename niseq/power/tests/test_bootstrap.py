from ..bootstrap import _bootstrap
import numpy as np

def test_bootstrap(mu = 10, std = 3, n = 100, n_boot = 1000, tol = .2):

    np.random.seed(0)

    def mean_stat(x, seed):
        return np.mean(x)
    def std_stat(x, seed):
        return np.std(x)

    x = np.random.normal(mu, std, n)
    mu_post = _bootstrap(x, mean_stat, n, n_boot)
    std_post = _bootstrap(x, std_stat, n, n_boot)

    assert(np.isclose(mu, mu_post.mean(), atol = tol))
    assert(np.isclose(std, std_post.mean(), atol = tol))

    return None

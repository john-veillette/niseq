from mne.utils import check_random_state
from ..cluster_test import (
    sequential_cluster_test_1samp,
    sequential_cluster_test_indep,
    sequential_cluster_test_corr
)
import numpy as np

def test_sequential_cluster_test_1samp(n = 30, seed = 0):
    rng = check_random_state(seed)
    X = rng.rand(n, 40, 40)
    look_times = np.linspace(n//3, n, 3).astype(int)
    for tail in [0, 1, -1]:
        # make sure function works
        out = sequential_cluster_test_1samp(X, look_times, n, tail = tail)
        # check output dimensions
        obs_stats, ps, adj_alpha, spending = out
        assert(adj_alpha.size == spending.size)
        assert(adj_alpha.size == ps.size)
        assert(X.shape[1:] == obs_stats[look_times[0]][0].shape)
    # test TFCE
    out = sequential_cluster_test_1samp(
        X, look_times, n,
        threshold = dict(start = 0, step = .1),
    )

def test_sequential_cluster_test_indep(n = 30, seed = 0):
    rng = check_random_state(seed)
    X = rng.rand(n, 40, 40)
    conds = rng.choice([0, 1], n)
    look_times = np.linspace(n//3, n, 3).astype(int)
    for tail in [0, 1, -1]:
        out = sequential_cluster_test_indep(X, conds, look_times, n, tail = tail)
        obs_stats, ps, adj_alpha, spending = out
        assert(adj_alpha.size == spending.size)
        assert(adj_alpha.size == ps.size)
        assert(X.shape[1:] == obs_stats[look_times[0]][0].shape)

def test_sequential_cluster_test_corr(n = 30, seed = 0):
    rng = check_random_state(seed)
    X = rng.rand(n, 40, 40)
    y = rng.rand(n)
    look_times = np.linspace(n//3, n, 3).astype(int)
    for tail in [0, 1, -1]:
        out = sequential_cluster_test_corr(X, y, look_times, n, tail = tail)
        obs_stats, ps, adj_alpha, spending = out
        assert(adj_alpha.size == spending.size)
        assert(adj_alpha.size == ps.size)
        assert(X.shape[1:] == obs_stats[look_times[0]][0].shape)

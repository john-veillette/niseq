################################################################################
# User-facing wrappers for low.level functions in ._permutation and ._clustering
################################################################################
from ._permutation import generate_permutation_dist, find_thresholds
from ._clustering import _get_cluster_stats, _get_cluster_pvs
from ._permutation import _get_cluster_stats_correlation
import numpy as np

def sequential_cluster_test_1samp(X, look_times, n_max, tail = 0, **kwargs):
    assert(isinstance(X, np.ndarray))
    obs, H0 = generate_permutation_dist(X, None, look_times, tail = tail, **kwargs)
    spending, adj_alpha = find_thresholds(H0, look_times, n_max, tail = tail)
    obs_stats, ps = _get_cluster_pvs(obs, H0, tail = tail)
    where_rejected = np.where(ps <= adj_alpha)[0]
    if where_rejected.size > 0:
        print('rejected null by sample size %d'%(look_times[0]))
    else:
        print('failed to reject null')
    return obs_stats, ps, adj_alpha, spending

def sequential_cluster_test_indep(X, y, look_times, n_max, tail = 0, **kwargs):
    assert(isinstance(X, list) or isinstance(X, tuple))
    obs, H0 = generate_permutation_dist(X, y, look_times, tail = tail, **kwargs)
    spending, adj_alpha = find_thresholds(H0, look_times, n_max, tail = tail)
    obs_stats, ps = _get_cluster_pvs(obs, H0, tail = tail)
    where_rejected = np.where(ps <= adj_alpha)[0]
    if where_rejected.size > 0:
        print('rejected null by sample size %d'%(look_times[0]))
    else:
        print('failed to reject null')
    return obs_stats, ps, adj_alpha, spending

def sequential_cluster_test_corr(X, y, look_times, n_max, tail = 0, **kwargs):
    assert(isinstance(X, np.ndarray) and isinstance(y, np.ndarray))
    obs, H0 = generate_permutation_dist(
        X, y, look_times,
        tail = tail,
        statistic = _get_cluster_stats_correlation,
        **kwargs
    )
    spending, adj_alpha = find_thresholds(H0, look_times, n_max, tail = tail)
    obs_stats, ps = _get_cluster_pvs(obs, H0, tail = tail)
    where_rejected = np.where(ps <= adj_alpha)[0]
    if where_rejected.size > 0:
        print('rejected null by sample size %d'%(look_times[0]))
    else:
        print('failed to reject null')
    return obs_stats, ps, adj_alpha, spending

from ._permutation import generate_permutation_dist, find_thresholds
from mne.stats.parametric import f_oneway, ttest_1samp_no_p, ttest_ind_no_p
from mne.stats.cluster_level import _pval_from_histogram
from mne.utils import check_random_state
from .util import _correlation_stat_fun
import numpy as np


def _format_input(X):
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in X]
    sample_shape = X[0][0].shape
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]
    return X, sample_shape


def _compute_max_stat(t_obs, tail):
    if tail == 0:
        max_stat = np.max(np.abs(t_obs))
    elif tail == 1:
        max_stat = np.max(t_obs)
    elif tail == -1:
        max_stat = np.min(t_obs)
    return max_stat


def _get_max_stat_samples(X, labels = None,
                        statfun = None, tail = 0, **statfun_kwargs):
    '''
    Computes observed univariate stats using statfun for observed data.

    Inputs
    -------
    X: an (n_samples, <sample_shape>) np.ndarray
    labels: for independent samples test, an (n_samples,) np.ndarray containing
            condition labels, otherwise None for one-sample test
    statfun: a function to compute test statistics from X
    tail: sidedness of test... 0, 1, or -1

    Returns
    -------
    t_obs: a (<sample_shape>) np.ndarray containing test statistics
    t_max: the maximum test statistic
    '''
    assert(isinstance(X, np.ndarray))
    if statfun is None:
        raise Exception('must specify statfun!')
    if labels is None:
        X = [X]
    else:
        conds = np.unique(labels)
        X = [X[labels == cond] for cond in conds]
    X, sample_shape = _format_input(X)
    t_obs = statfun(*X, **statfun_kwargs)
    max_stat = _compute_max_stat(t_obs, tail)
    return t_obs.reshape(sample_shape), max_stat


def _get_max_stat_correlation(X, labels, statfun, tail = 0, **statfun_kwargs):
    '''
    Inputs
    -------
    X: an (n_samples, <sample_shape>) np.ndarray
    y: an (n_samples,) np.ndarray
    statfun: a function to compute test statistics from X and y
    tail: sidedness of test... 0, 1, or -1

    Returns
    -------
    r_obs: a (<sample_shape>) np.ndarray containing test statistics
    r_max: the maximum test statistic
    '''
    X = [X, y]
    X, sample_shape = _format_input(X)
    r_obs = statfun(*X, **statfun_kwargs)
    max_stat = _compute_max_stat(r_obs, tail)
    return r_obs.reshape(sample_shape), max_stat


def _get_pvs_from_histogram(T_obs, H0, tail = 0):
    obs_shape = T_obs.shape
    T_obs = T_obs.flatten()
    if tail == 0:
        p_values = (H0 >= np.abs(T_obs[:, np.newaxis])).mean(-1)
    elif tail == 1:
        p_values = (H0 >= T_obs[:, np.newaxis]).mean(-1)
    elif tail == -1:
        p_values = (H0 <= T_obs[:, np.newaxis]).mean(-1)
    return p_values.reshape(obs_shape)


def _add_pvs_to_obs(obs_stats, H0, tail):
    '''
    formats observed data and associated p-values in similar format to
    mne.stats.permutations.permutation_t_test for each look time in obs_stats
    '''
    formatted_obs = {}
    min_ps = []
    for i, look_time in enumerate(obs_stats):
        obs = obs_stats[look_time][0]
        h0 = H0[:, i]
        p = _get_pvs_from_histogram(obs, h0, tail)
        formatted_obs[look_time] = (obs, p, h0)
        min_ps.append(np.min(p))
    return formatted_obs, np.array(min_ps)


def sequential_permutation_t_test_1samp(X,
                                        look_times, n_max,
                                        tail = 0, **kwargs):
    '''
    sequential generalization of mne.stats.permutations.permutation_t_test
    '''
    assert(isinstance(X, np.ndarray))
    obs, H0 = generate_permutation_dist(
        X, None,
        look_times,
        tail = tail,
        statistic = _get_max_stat_samples,
        statfun = ttest_1samp_no_p,
        **kwargs
        )
    spending, adj_alpha = find_thresholds(H0, look_times, n_max, tail = tail)
    obs_stats, ps = _add_pvs_to_obs(obs, H0, tail)
    return obs_stats, ps, adj_alpha, spending


def sequential_permutation_test_indep(X, labels,
                                        look_times, n_max,
                                        tail = 0, **kwargs):
    '''
    sequential generalization of mne.stats.permutations.permutation_t_test

    Uses either a t statistic for two groups or a one-way F statistic
    for multiple groups.
    '''
    assert(isinstance(X, np.ndarray))
    assert(isinstance(labels, np.ndarray))
    if np.unique(labels).size == 2:
        sf = ttest_ind_no_p
    else:
        sf = f_oneway
    obs, H0 = generate_permutation_dist(
        X, labels,
        look_times,
        tail = tail,
        statistic = _get_max_stat_samples,
        statfun = sf,
        **kwargs
        )
    spending, adj_alpha = find_thresholds(H0, look_times, n_max, tail = tail)
    obs_stats, ps = _add_pvs_to_obs(obs, H0, tail)
    return obs_stats, ps, adj_alpha, spending


def sequential_permutation_test_corr(X, y,
                                    look_times, n_max,
                                    tail = 0, **kwargs):
    '''
    sequential permutation test with correlation as the univariate test statistic
    '''
    assert(isinstance(X, np.ndarray))
    assert(isinstance(y, np.ndarray))
    obs, H0 = generate_permutation_dist(
        X, y,
        look_times,
        tail = tail,
        statistic = _get_max_stat_correlation,
        statfun = _correlation_stat_fun,
        **kwargs
        )
    spending, adj_alpha = find_thresholds(H0, look_times, n_max, tail = tail)
    obs_stats, ps = _add_pvs_to_obs(obs, H0, tail)
    return obs_stats, ps, adj_alpha, spending

# -*- coding: utf-8 -*-

# Authors: John Veillette <johnv@uchicago.edu>
#
# License: BSD-3-Clause

from ._permutation import generate_permutation_dist, find_thresholds
from mne.stats.parametric import f_oneway, ttest_1samp_no_p, ttest_ind_no_p
from mne.stats.cluster_level import _pval_from_histogram
from mne.utils import check_random_state, warn
from .util import _correlation_stat_fun
import numpy as np

from .util.docs import fill_doc

def _format_input(X):
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in X]
    sample_shape = X[0][0].shape
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]
    return X, sample_shape


def _compute_max_stat(t_obs, tail):
    if tail == 0:
        max_stat = np.nanmax(np.abs(t_obs))
    elif tail == 1:
        max_stat = np.nanmax(t_obs)
    elif tail == -1:
        max_stat = np.nanmin(t_obs)
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


def _get_max_stat_correlation(X, y, statfun, tail = 0, **statfun_kwargs):
    '''
    Inputs
    -------
    X: an (n_samples, <sample_shape>) np.ndarray
    y: an (n_samples,) np.ndarray
    statfun: a function to compute test statistics from X and y
    tail: sidedness of test... 0, 1, or -1

    Returns
    -------
    r_obs : a (<sample_shape>) np.ndarray containing test statistics
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
        # p should never be zero in a permutation test, so if p == 0, then
        p[p == 0] = np.nan # stat must have been compared to NaN or Inf
        formatted_obs[look_time] = (obs, p, h0)
        min_ps.append(np.nanmin(p))
    return formatted_obs, np.array(min_ps)

@fill_doc
def sequential_permutation_t_test_1samp(X,
                                        look_times, n_max,
                                        alpha = .05, tail = 0,
                                        spending_func = None,
                                        verbose = True,
                                        **kwargs):
    '''One-sample sequential permutation test with max-type correction.

    This is a sequential generalization of
    ``mne.stats.permutations.permutation_t_test``.

    Uses max-type correction for multiple comparisons [4].

    %(alpha_spending_explanation)s

    Parameters
    ----------
    %(X)s
    %(look_times)s
    %(n_max)s
    %(alpha)s
    %(tail)s
    %(spending_func)s
    %(verbose)s
    %(n_permutations)s
    %(n_jobs)s
    %(seed)s

    Returns
    ---------
    %(returns_maxtype)s

    Notes
    ---------
    %(alpha_spending_note)s

    References
    ----------
    %(references_maxtype)s





    '''
    assert(isinstance(X, np.ndarray))
    obs, H0 = generate_permutation_dist(
        X, None,
        look_times,
        tail = tail,
        statistic = _get_max_stat_samples,
        statfun = ttest_1samp_no_p,
        verbose = verbose,
        **kwargs
        )
    spending, adj_alpha = find_thresholds(
        H0, look_times, n_max,
        alpha, tail, spending_func
        )
    obs_stats, ps = _add_pvs_to_obs(obs, H0, tail)
    return obs_stats, ps, adj_alpha, spending


@fill_doc
def sequential_permutation_test_indep(X, labels,
                                        look_times, n_max,
                                        alpha = .05, tail = 0,
                                        spending_func = None,
                                        verbose = True,
                                        **kwargs):
    '''Independent-sample sequential permutation test with max-type correction.

    By default, this is a sequential generalization of an independent sample
    max-t procedure if two groups and max-F procedure if more groups.

    Uses max-type correction for multiple comparisons [4].

    %(alpha_spending_explanation)s

    Parameters
    ----------
    %(X)s
    %(labels)s
    %(look_times)s
    %(n_max)s
    %(alpha)s
    %(tail)s
    %(spending_func)s
    %(verbose)s
    %(n_permutations)s
    %(n_jobs)s
    %(seed)s

    Returns
    ---------
    %(returns_maxtype)s

    Notes
    ---------
    %(alpha_spending_note)s

    References
    ----------
    %(references_maxtype)s





    '''
    assert(isinstance(X, np.ndarray))
    assert(isinstance(labels, np.ndarray))
    if np.unique(labels).size == 2:
        sf = ttest_ind_no_p
    else:
        sf = f_oneway
        if tail != 1:
            warn('Ignoring argument "tail", performing 1-tailed F-test')
            tail = 1

    obs, H0 = generate_permutation_dist(
        X, labels,
        look_times,
        tail = tail,
        statistic = _get_max_stat_samples,
        statfun = sf,
        verbose = verbose,
        **kwargs
        )
    spending, adj_alpha = find_thresholds(
        H0, look_times, n_max,
        alpha, tail, spending_func
        )
    obs_stats, ps = _add_pvs_to_obs(obs, H0, tail)
    return obs_stats, ps, adj_alpha, spending


@fill_doc
def sequential_permutation_test_corr(X, y,
                                    look_times, n_max,
                                    alpha = .05, tail = 0,
                                    spending_func = None,
                                    verbose = True,
                                    **kwargs):
    '''A sequential permutation test for correlations with a max-type correction.

    Tests for a relationship between ``X`` and a continuous independent variable
    ``y``. By default, uses Pearson correlation by default, but the test
    statistic can be modified.

    Uses max-type correction for multiple comparisons [4].

    %(alpha_spending_explanation)s

    Parameters
    ----------
    %(X)s
    %(y)s
    %(look_times)s
    %(n_max)s
    %(alpha)s
    %(tail_corr)s
    %(spending_func)s
    %(verbose)s
    %(n_permutations)s
    %(n_jobs)s
    %(seed)s

    Returns
    ---------
    %(returns_maxtype)s

    Notes
    ---------
    %(alpha_spending_note)s

    References
    ----------
    %(references_maxtype)s





    '''
    assert(isinstance(X, np.ndarray))
    assert(isinstance(y, np.ndarray))
    obs, H0 = generate_permutation_dist(
        X, y,
        look_times,
        tail = tail,
        statistic = _get_max_stat_correlation,
        statfun = _correlation_stat_fun,
        verbose = verbose,
        **kwargs
        )
    spending, adj_alpha = find_thresholds(
        H0, look_times, n_max,
        alpha, tail, spending_func
        )
    obs_stats, ps = _add_pvs_to_obs(obs, H0, tail)
    return obs_stats, ps, adj_alpha, spending

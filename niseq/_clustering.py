# -*- coding: utf-8 -*-

# Authors: John Veillette <johnv@uchicago.edu>
#
# License: BSD-3-Clause

from mne.stats.parametric import f_oneway, ttest_1samp_no_p
from mne.stats.cluster_level import (
    _permutation_cluster_test,
    _setup_adjacency,
    _cluster_indices_to_mask,
    _cluster_mask_to_indices,
    _get_partitions_from_adjacency,
    _validate_type,
    _find_clusters,
    _reshape_clusters,
    _pval_from_histogram
)
from mne.utils import _check_option, warn
from .util import _correlation_stat_fun
from collections import OrderedDict
from scipy.stats import beta
import numpy as np

def _check_fun(X, stat_fun, threshold, tail = 0, kind = 'within'):
    '''
    Same as mne.stats.cluster_level._check_fun but less verbose
    '''
    from scipy import stats
    if kind == 'within':
        ppf = stats.t.ppf
        if threshold is None:
            if stat_fun is not None and stat_fun is not ttest_1samp_no_p:
                warn('Automatic threshold is only valid for stat_fun=None '
                     '(or ttest_1samp_no_p), got %s' % (stat_fun,))
            p_thresh = 0.05 / (1 + (tail == 0))
            n_samples = len(X)
            threshold = -ppf(p_thresh, n_samples - 1)
            if np.sign(tail) < 0:
                threshold = -threshold
        stat_fun = ttest_1samp_no_p if stat_fun is None else stat_fun
    else:
        assert kind == 'between'
        ppf = stats.f.ppf
        if threshold is None:
            if stat_fun is not None and stat_fun is not f_oneway:
                warn('Automatic threshold is only valid for stat_fun=None '
                     '(or f_oneway), got %s' % (stat_fun,))
            elif tail != 1:
                warn('Ignoring argument "tail", performing 1-tailed F-test')
                tail = 1
            p_thresh = 0.05
            dfn = len(X) - 1
            dfd = np.sum([len(x) for x in X]) - len(X)
            threshold = ppf(1. - p_thresh, dfn, dfd)
        stat_fun = f_oneway if stat_fun is None else stat_fun
    return stat_fun, threshold, tail

def _get_cluster_stats(X, threshold = None, max_step = 1,
        tail = 0, stat_fun = None, adjacency = None,
        exclude = None, t_power = 1,
        out_type = 'mask', check_disjoint = False):
    '''
    An auxilliary function to compute cluster statistics from observed data or
    from a single permutation

    mostly copied from mne.stats.cluster_level._permutation_cluster_test, but
    it doesn't perform a full permutation test, just gets the observed stats
    '''
    n_samples = X[0].shape[0]
    n_times = X[0].shape[1]
    sample_shape = X[0].shape[1:]

    _check_option('out_type', out_type, ['mask', 'indices'])
    _check_option('tail', tail, [-1, 0, 1])
    if not isinstance(threshold, dict):
        threshold = float(threshold)
        if (tail < 0 and threshold > 0 or tail > 0 and threshold < 0 or
                tail == 0 and threshold < 0):
            raise ValueError('incompatible tail and threshold signs, got '
                             '%s and %s' % (tail, threshold))

    # flatten the last dimensions in case the data is high dimensional
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]
    n_tests = X[0].shape[1]

    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    if (exclude is not None) and not exclude.size == n_tests:
        raise ValueError('exclude must be the same shape as X[0]')

    t_obs = stat_fun(*X)
    _validate_type(t_obs, np.ndarray, 'return value of stat_fun')

    ## Compute mass-univariate test stat
    #--------------------------------------
    if t_obs.size != np.prod(sample_shape):
        raise ValueError('t_obs.shape %s provided by stat_fun %s is not '
                         'compatible with the sample shape %s'
                         % (t_obs.shape, stat_fun, sample_shape))
    if adjacency is None or adjacency is False:
        t_obs.shape = sample_shape

    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    ## compute cluster statistic from univariate test results
    # ------------------------------------------------------
    # determine if adjacency itself can be separated into disjoint sets
    if check_disjoint is True and (adjacency is not None and
                                   adjacency is not False):
        partitions = _get_partitions_from_adjacency(adjacency, n_times)
    else:
        partitions = None
    out = _find_clusters(t_obs, threshold, tail, adjacency,
                         max_step = max_step, include = include,
                         partitions = partitions, t_power = t_power,
                         show_info = False)
    clusters, cluster_stats = out

    # The stat should have the same shape as the samples
    t_obs.shape = sample_shape

    # For TFCE, return the "adjusted" statistic instead of raw scores
    if isinstance(threshold, dict):
        t_obs = cluster_stats.reshape(t_obs.shape) * np.sign(t_obs)
        cluster_stats = t_obs.reshape(cluster_stats.shape)

    # convert clusters to old format
    if adjacency is not None and adjacency is not False:
        # our algorithms output lists of indices by default
        if out_type == 'mask':
            clusters = _cluster_indices_to_mask(clusters, n_tests)
    else:
        # ndimage outputs slices or boolean masks by default
        if out_type == 'indices':
            clusters = _cluster_mask_to_indices(clusters, t_obs.shape)

    clusters = _reshape_clusters(clusters, sample_shape)
    if len(clusters) == 0: # handle case in which no clusters are found
        cluster_stats = np.array([0])
    if tail == 0:
        max_stat = np.max(np.abs(cluster_stats))
    elif tail == 1:
        max_stat = np.max(cluster_stats)
    elif tail == -1:
        max_stat = np.min(cluster_stats)
    return t_obs, clusters, cluster_stats, max_stat


def _get_cluster_stats_samples(X, labels = None, threshold = None, max_step = 1,
        tail = 0, stat_fun = None, adjacency = None,
        exclude = None, t_power = 1,
        out_type = 'mask', check_disjoint = False):
    '''
    Computes cluster stats when design is one-sample or independent samples
    '''
    ##  check inputs
    # -------------------------
    assert(isinstance(X, np.ndarray))
    if labels is None:
        X = [X]
        stat_fun, threshold, tail = _check_fun(X[0], stat_fun, threshold, tail)
    else:
        conds = np.unique(labels)
        X = [X[labels == cond] for cond in conds]
        stat_fun, threshold, tail = _check_fun(X, stat_fun, threshold, tail, 'between')

    # check dimensions for each group in X (a list at this stage).
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in X]
    sample_shape = X[0].shape[1:]
    for x in X:
        if x.shape[1:] != sample_shape:
            raise ValueError('All samples mush have the same size')

    t_obs, clusters, cluster_stats, max_stat = _get_cluster_stats(
            X, threshold, max_step,
            tail, stat_fun, adjacency,
            exclude , t_power,
            out_type, check_disjoint)
    return t_obs, clusters, cluster_stats, tail, max_stat


def _correlation_stat_fun_tfce(X, y):
    r = _correlation_stat_fun(X, y)
    z = np.arctanh(r) # Fisher z-transform
    return z

def _check_fun_correlation(X, stat_fun, threshold, tail):
    n = X.shape[0]
    if stat_fun is None:
        if isinstance(threshold, dict): # use z-transformed correlations
            stat_fun = _correlation_stat_fun_tfce
        else:
            stat_fun = _correlation_stat_fun
            if threshold is None:
                # default to r value where p < 0.05
                alpha = .05 / (1 + (tail == 0))
                dist = beta(n/2 - 1, n/2 - 1, loc = -1, scale = 2)
                if tail == 0:
                    threshold = -dist.ppf(alpha / 2)
                elif tail == 1:
                    threshold = -dist.ppf(alpha)
                elif tail == -1:
                    threshold = dist.ppf(alpha)
    else:
        assert(threshold is not None) # if stat_fun given, must define threshold
    return stat_fun, threshold

def _get_cluster_stats_correlation(X, y, threshold = None, max_step = 1,
        tail = 0, stat_fun = None, adjacency = None,
        exclude = None, t_power = 1,
        out_type = 'mask', check_disjoint = False):
    '''
    Computes cluster stats when y is a regression outcome variable
    '''
    assert(isinstance(X, np.ndarray) and isinstance(y, np.ndarray))
    assert(y.ndim == 1)
    stat_fun, threshold = _check_fun_correlation(X, stat_fun, threshold, tail)
    return _get_cluster_stats([X, y], threshold, max_step,
            tail, stat_fun, adjacency,
            exclude , t_power,
            out_type, check_disjoint)


def _get_cluster_pvs(obs_stats, H0, tail):
    '''
    computes cluster p-values at each look time by comparing observed clusters
    to permutation distribution

    Inputs
    --------
    obs_stats: (dict) of tuples of (t_obs, clusters, cluster_stats)
                indexed by look time
    H0: (np.ndarray) joint null distribution of cluster stats across look times
    tail: (int) sidedness of test to compute p-value for

    Returns
    --------
    stats: (dict of tuples) indexed by look time, this dict contains output
            matching that of MNE's cluster-based permutation test for that
            look time (i.e. t_obs, clusters, cluster_ps, H0)
    p_values: (np.ndarray) smallest p-value at each look time
    '''
    obs = OrderedDict()
    min_ps = [] # minimum at each look time
    for i, n in enumerate(obs_stats):
        t_obs, clusters, cluster_stats = obs_stats[n][:3]
        h0 = H0[:, i]
        if len(clusters) > 0:
            cluster_pv = _pval_from_histogram(cluster_stats, h0, tail)
            min_ps.append(np.min(cluster_pv))
        else:
            cluster_pv = np.array([])
            min_ps.append(1.) # no clusters
        obs[n] = (t_obs, clusters, cluster_pv, h0)
    return obs, np.array(min_ps)

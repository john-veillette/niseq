# -*- coding: utf-8 -*-

# Authors: John Veillette <johnv@uchicago.edu>
#
# License: BSD-3-Clause

from ._permutation import generate_permutation_dist, find_thresholds
from ._clustering import _get_cluster_stats, _get_cluster_pvs
from ._permutation import _get_cluster_stats_correlation
import numpy as np

from .util.docs import fill_doc

@fill_doc
def sequential_cluster_test_1samp(X, look_times, n_max, alpha = .05, tail = 0,
                                    spending_func = None,
                                    verbose = True, **kwargs):
    '''A sequential one-sample cluster test.

    A sequential generalization of a one-sample cluster-based permutation test
    (as described by [4]) or of TFCE (as described by [6]).

    %(alpha_spending_explanation)s

    Parameters
    ----------
    %(X_clust)s
    %(look_times)s
    %(n_max)s
    %(alpha)s
    %(tail)s
    %(spending_func)s
    %(verbose)s
    %(threshold)s
    %(n_permutations)s
    %(stat_fun_clust_t)s
    %(adjacency_clust_1)s
    %(n_jobs)s
    %(seed)s
    %(max_step_clust)s
    %(exclude_clust)s
    %(t_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s

    Returns
    ---------
    %(returns_clust)s

    Notes
    ---------
    %(alpha_spending_note)s

    References
    ----------
    %(references_clust)s




    '''
    assert(isinstance(X, np.ndarray))
    obs, H0 = generate_permutation_dist(
        X, None,
        look_times,
        tail = tail,
        verbose = verbose,
        **kwargs)
    spending, adj_alpha = find_thresholds(
        H0, look_times, n_max,
        alpha, tail, spending_func
        )
    obs_stats, ps = _get_cluster_pvs(obs, H0, tail = tail)
    return obs_stats, ps, adj_alpha, spending


@fill_doc
def sequential_cluster_test_indep(X, labels, look_times, n_max, alpha = .05,
                                tail = 0, spending_func = None,
                                verbose = True, **kwargs):
    '''A sequential independent-sample cluster test.

    A sequential generalization of an independet-sample cluster-based
    permutation test (as described by [4]) or of TFCE (as described by [6]).

    %(alpha_spending_explanation)s

    Parameters
    ----------
    %(X_clust)s
    %(labels)s
    %(look_times)s
    %(n_max)s
    %(alpha)s
    %(tail_clust)s
    %(spending_func)s
    %(verbose)s
    %(threshold)s
    %(n_permutations)s
    %(tail)s
    %(stat_fun_clust_f)s
    %(adjacency_clust_1)s
    %(n_jobs)s
    %(seed)s
    %(max_step_clust)s
    %(exclude_clust)s
    %(f_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s

    Returns
    ---------
    %(returns_clust)s

    Notes
    ---------
    %(alpha_spending_note)s

    References
    ----------
    %(references_clust)s




    '''
    assert(isinstance(X, np.ndarray))
    assert(isinstance(X, np.ndarray))
    obs, H0 = generate_permutation_dist(
        X, labels, look_times,
        tail = tail,
        verbose = verbose,
        **kwargs
    )
    spending, adj_alpha = find_thresholds(
        H0, look_times, n_max,
        alpha, tail, spending_func
        )
    tail = obs[look_times[0]][-2] # in case tail was overridden
    obs_stats, ps = _get_cluster_pvs(obs, H0, tail = tail)
    return obs_stats, ps, adj_alpha, spending


@fill_doc
def sequential_cluster_test_corr(X, y, look_times, n_max, alpha = .05, tail = 0,
                                    spending_func = None,
                                    verbose = True, **kwargs):
    '''A sequential cluster test for correlations.

    A sequential generalization of a cluster-based permutation test
    (as described by [4]) or of TFCE (as described by [6]) for testing a
    relationship between ``X`` and a continuous variable ``y``. Uses Pearson
    correlation by default (or its z-transform if using TFCE), but test
    statistic can be modified.

    %(alpha_spending_explanation)s

    Parameters
    ----------
    %(X_clust)s
    %(y)s
    %(look_times)s
    %(n_max)s
    %(alpha)s
    %(tail)s
    %(spending_func)s
    %(verbose)s
    %(threshold)s
    %(n_permutations)s
    %(tail_corr)s
    %(stat_fun_corr)s
    %(adjacency_clust_1)s
    %(n_jobs)s
    %(seed)s
    %(max_step_clust)s
    %(exclude_clust)s
    %(f_power_clust)s
    %(out_type_clust)s
    %(check_disjoint_clust)s

    Returns
    ---------
    %(returns_clust)s

    Notes
    ---------
    %(alpha_spending_note)s

    References
    ----------
    %(references_clust)s




    '''
    assert(isinstance(X, np.ndarray) and isinstance(y, np.ndarray))
    obs, H0 = generate_permutation_dist(
        X, y, look_times,
        tail = tail,
        statistic = _get_cluster_stats_correlation,
        verbose = verbose,
        **kwargs
    )
    spending, adj_alpha = find_thresholds(
        H0, look_times, n_max,
        alpha, tail, spending_func
        )
    obs_stats, ps = _get_cluster_pvs(obs, H0, tail = tail)
    return obs_stats, ps, adj_alpha, spending

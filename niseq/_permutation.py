from ._clustering import _get_cluster_stats_samples, _get_cluster_stats_correlation
from .spending_functions import SpendingFunction, LinearSpendingFunction
from mne.utils import check_random_state
from mne.parallel import parallel_func
from collections import OrderedDict
import numpy as np

def _get_stat_at_look_times(X, labels, look_times, statistic, statistic_kwargs):
    '''
    computes the test statistic at each look time
    '''
    obs = OrderedDict()

    for i, n in enumerate(look_times):

        _X = X[:n]

        if labels is None: # one-sample test
            obs[n] = statistic(_X, **statistic_kwargs)
        else: # independent-samples test
            _labels = labels[:n]
            obs[n] = statistic(_X, _labels, **statistic_kwargs)

    return obs


def _get_stat_perm(X, labels, look_times, seed, statistic, statistic_kwargs):
    '''
    gets the test statistics, at each look time, for a single permutation
    '''
    rng = check_random_state(seed)

    if labels is None: # one-sample, permute by sign flips
        flips = rng.choice([-1, 1], size = X.shape[0])
        while X.ndim > flips.ndim:
            flips = flips[:, np.newaxis]
        _X = X * flips
        _labels = None
    else: # independent-sample, permute by shuffling labels:
        # but only shuffle within each period between look times,
        # so p-values at each look time will be approximately preserved
        # across multiple, sequential uses of the test
        idxs = np.arange(X.shape[0])
        _lt = [0] + list(look_times)
        starts = [_lt[i] for i in range(len(look_times))]
        ends = [_lt[i + 1] for i in range(len(look_times))]
        for s, e in zip(starts, ends):
            rng.shuffle(idxs[s:e])
        _X = X
        _labels = labels[idxs]
    stats = _get_stat_at_look_times(
        _X, _labels,
        look_times,
        statistic, statistic_kwargs
    )
    H0 = [stats[n] for n in look_times]
    if isinstance(H0[0], tuple):
        H0 = [h0[-1] for h0 in H0]
    H0 = np.array(H0)
    assert(H0.ndim == 1)
    return H0


def generate_permutation_dist(X, labels,
                            look_times,
                            n_permutations = 1024,
                            seed = None,
                            n_jobs = 1,
                            statistic = _get_cluster_stats_samples,
                            **statistic_kwargs):
    '''
    computes test statistic and its permutation distribution at each look time
    '''
    # pick new random seeds to use for each permutation
    rng = check_random_state(seed)
    perm_seeds = rng.randint(0, n_permutations * 1000, n_permutations)

    obs = _get_stat_at_look_times(
        X, labels,
        look_times,
        statistic, statistic_kwargs
    )
    obs_stats = [obs[n] for n in look_times]
    if isinstance(obs_stats[0], tuple): # test stat should be last
        obs_stats = [s[-1] for s in obs_stats]
    obs_stats = np.array(obs_stats)

    parallel, p_func, n_jobs = parallel_func(_get_stat_perm, n_jobs)
    out = parallel(
        p_func(
            X, labels,
            look_times,
            ps,
            statistic, statistic_kwargs
        ) for ps in perm_seeds
    )
    H0 = np.stack(out)
    H0 = np.concatenate([obs_stats[np.newaxis, :], H0])
    return obs, H0


def _quantile(x, q, method):
    '''
    deal with older numpy versions that only use deprecated `interpolation` arg
    '''
    try:
        return np.quantile(x, q, method = method)
    except:
        return np.quantile(x, q, interpolation = method)

def _throw_spending_func_warning():
    '''
    raised when user spending function doesn't match specified alpha and n_max
    '''
    import warnings
    warnings.warn(
        '''
        User input spending_func doesn't match `alpha` and `n_max` arguments!
        `alpha` and `n_max` have been overridden by those of `spending_func`.
        '''
    )

def find_thresholds(
    H0, look_times, max_n,
    alpha = 0.05, tail = 0,
    spending_func = None
    ):
    '''
    Given a permutation null distribution for a corresponding sequence of look
    times and an alpha spending function, computes the adjusted significance
    thresholds requires to control the false positive rate across all looks.
    '''
    # check spending function
    if spending_func is None:
        spending_func = LinearSpendingFunction(alpha, max_n)
    else:
        assert(isinstance(spending_func, SpendingFunction))
        try:
            assert(spending_func.alpha == alpha)
            assert(spending_func.max_n == max_n)
        except:
            alpha = spending_func.alpha
            max_n = spending_func.max_n
            _throw_spending_func_warning()

    _H0 = np.copy(H0)
    spending_hist = []
    adjusted_alphas = []
    assert(isinstance(spending_func, SpendingFunction))
    for i, n in enumerate(look_times):
        budget = spending_func(n)
        if tail == 1:
            thres = _quantile(_H0[:, i], 1 - budget, 'higher')
            adjusted_alpha = np.mean(H0[:, i] > thres)
            # If a permutation exceeds threshold at one look time...
            perm_false_positives = _H0[:, i] >= thres
            # ... we pass it forward so counted against future looks
            _H0[perm_false_positives, :] = np.inf
        elif tail == -1:
            thres = _quantile(_H0[:, i], budget, 'lower')
            adjusted_alpha = np.mean(H0[:, i] < thres)
            perm_false_positives = _H0[:, i] <= thres
            _H0[perm_false_positives, :] = -np.inf
        else:
            thres = _quantile(np.abs(_H0[:, i]),1 - budget, 'higher')
            adjusted_alpha = np.mean(np.abs(H0[:, i]) > thres)
            perm_false_positives = np.abs(_H0[:, i]) >= thres
            _H0[perm_false_positives, :] = np.inf
        spending_hist.append(budget)
        adjusted_alphas.append(adjusted_alpha)
    return np.array(spending_hist), np.array(adjusted_alphas)

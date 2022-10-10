from .spending_functions import SpendingFunction, LinearSpendingFunction
from ._clustering import _get_cluster_stats
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
            _X = [_X]
        else: # independent-samples test
            _labels = labels[:n]
            conds = np.unique(_labels)
            _X = [_X[_labels == cond] for cond in conds]
        obs[n] = statistic(_X, **statistic_kwargs)

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
    else: # independent-sample, permute by shuffle
        idxs = np.arange(X.shape[0])
        rng.shuffle(idxs)
        _X = X[idxs]
        _labels = labels
    stats = _get_stat_at_look_times(
        _X, _labels,
        look_times,
        statistic, statistic_kwargs
    )
    H0 = [stats[n] for n in look_times]
    if isinstance(H0[0], tuple): # if statistic() returns multiple values,
        H0 = [np.max(h0[-1]) for h0 in H0] # then test stats should be last
    H0 = np.array(H0)
    assert(H0.ndim == 1)
    return H0


def generate_permutation_dist(X, labels,
                            look_times,
                            n_permutations = 1024,
                            seed = None,
                            n_jobs = 1,
                            statistic = _get_cluster_stats,
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
    if isinstance(obs_stats[0], tuple):
        obs_stats = [np.max(s[-1]) for s in obs_stats]
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


def find_thresholds(
    H0, look_times, max_n,
    alpha = 0.05, tail = 0,
    spending_function = LinearSpendingFunction
    ):
    '''
    Given a permutation null distribution for a corresponding sequence of look
    times and an alpha spending function, computes the adjusted significance
    thresholds requires to control the false positive rate across all looks.
    '''
    _H0 = np.copy(H0)
    spending_hist = []
    adjusted_alphas = []
    spending_func = spending_function(alpha, max_n) # check is valid
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

# -*- coding: utf-8 -*-

# Authors: John Veillette <johnv@uchicago.edu>
#
# License: BSD-3-Clause


from ..spending_functions import SpendingFunction
from mne.parallel import parallel_func
from mne.utils import check_random_state
from mne import verbose

from inspect import isclass, isfunction
import numpy as np

from ..util.docs import fill_doc

def _boot_sample(x, n_resample, seed):
    '''
    performs two-level sampling of
        1. sample weights from direchlet distribution
        2. samples from x according to weights, with replacement

    Dirichlet distribution uses parameter alpha = 4, since this gives
    better approximations for the Bayesian bootstrap according
    to Tu and Zheng (1987).
    '''
    n_orig = x.shape[0]
    rng = check_random_state(seed)
    weights = rng.dirichlet(4 * np.ones(n_orig), 1)[0]
    idx = rng.choice(
        np.arange(n_orig), size = n_resample,
        replace = True, p = weights
    )
    x_boot = x[idx]
    return x_boot

def _boot_stat(x, statistic, n_resample, conditional = False, seed = 0, **kwargs):
    '''
    performs one bootstrap resampling and then computes statistic on new sample

    Equivalent to one draw of statistic from the posterior distribution
    '''
    if conditional:
        x_boot = _boot_sample(x, n_resample - x.shape[0], seed)
        _x = np.concatenate([x, x_boot])
    else:
        x_boot = _boot_sample(x, n_resample, seed)
        _x = x_boot
    kwargs['seed'] = seed
    return statistic(_x, **kwargs)

def _bootstrap(x, statistic, n_resample, n_simulations,
                conditional = False, seed = 0, n_jobs = None, **kwargs):
    '''
    performs full Bayesian bootstrap, optionally conditioned on current data
    '''
    # pick new random seeds to use for each permutation
    rng = check_random_state(seed)
    sim_seeds = rng.randint(1, np.iinfo(np.int32).max - 1, n_simulations)
    # perform `n_simulations` bootsrap resamplings
    parallel, p_func, _ = parallel_func(_boot_stat, n_jobs)
    out = parallel(
        p_func(x, statistic, n_resample, conditional, seed, **kwargs)
            for seed in sim_seeds
    )
    return np.stack(out)

_cumulative_power = lambda rejs: (np.cumsum(rejs, axis = 1) > 0).mean(0)

def _prob_rejection(rejs):
    assert(rejs.ndim == 2)
    first_rejs = np.zeros_like(rejs)
    for row in range(rejs.shape[0]):
        idxs = np.where(rejs[row, :])[0]
        if idxs.size > 0:
            idx = np.min(idxs)
            first_rejs[row, idx] = 1
    return first_rejs.mean(0)

def _expected_sample_size(rejs, look_times):
    idxs = [np.where(r)[0] for r in rejs]
    idxs = [np.min(i) if i.size > 0 else rejs.shape[1] - 1 for i in idxs]
    ns = [look_times[i] for i in idxs]
    return np.mean(ns)

@fill_doc
def bootstrap_predictive_power_1samp(X, test_func, look_times, n_max,
                alpha = .05, conditional = False, n_simulations = 1024,
                seed = None, n_jobs = None, **test_func_kwargs):
    '''Predictive power analysis via Bayesian bootstrap

    Computes the predictive power non-parametrically using the Bayesian
    bootstrap. Optionally, you can condition on the current data to get
    conditional power, which is useful for adaptive designs. Only valid for
    one-sample (or paired-sample) tests.

    Statistics computed from resamples using the Bayesian bootstrap, as opposed
    to the frequentist boostrap, can be interpreted as draws from the posterior
    distribution with an uninformative prior [1]. Thus, results here can be
    conveniently interpreted as the Bayesian predictive power. As recommended by
    [2] (not in English) and helpfully restated by [3] (in English), resampling
    weights are drawn from Dirichlet(alpha = 4) for a better approximation.

    This functionality is experimental. It is the best catch-all way to do a
    power analysis for permutation tests I can think of, and similar resampling
    approaches to estimating power have been used in the literature (e.g. by
    [4]); however, it should be noted that the neuroimaging literature has not
    converged upon a standardized approach to performing power analyses.
    The Bayesian bootstrap approach used here incorporates uncertainty about
    the effect size into the power estimate, which is handy since uncertainty
    about the true effect size is considerable following a small pilot study, or
    even a typical psychology/neuroimaging sample size, as pointed out by [5].

    Parameters
    -----------
    X : array, shape (n_observations, p[, q][, r])
        The data from which to resample. ``X`` should contain the observations
        for one group or paired differences. The first dimension of the array
        is the number of observations; remaining dimensions comprise the size of
        a single observation. See documentation for user-input ``test_func`` for
        more details.
    test_func : function
        The one-sample sequential test you want to run a power analysis for.
        Must accept ``look_times``, ``n_max``, ``alpha``, and ``verbose``
        arguments and return results, the middle two of which are the p-values
        for each look and the adjusted alphas, respectively. This could be any
        user-facing function from ``niseq`` that ends in ``_1samp``.
    %(look_times)s
    %(n_max)s
    %(alpha)s
    conditional : bool, default: ``False``
        If ``True``, performs a conditional power analysis; that is, computes
        the probability of a design rejecting the null hypothesis given that the
        data in ``X`` has already been collected and is included in the
        analysis, as in an adaptive design. If ``False`` (default), performs a
        prospective power analysis (e.g. if you're using pilot data or data from
        another study to inform sample size planning for a study that hasn't
        begun data collection).
    n_simulations: int, default: ``1024``
        Number of bootstrap resamples/simulations to perform.
    %(seed)s
    %(n_jobs)s
    **test_func_kwargs:
        You may input any arguments you'd like to be passed to ``test_func``.


    Returns
    --------
    res : dict
        A results dictionary with keys:

        ``'uncorr_instantaneous_power'`` : list of float
            The power of a fixed-sample statistical test performed at each look.
        ``'rejection_probability'`` : list of float
            The probability that a sequential test rejects the null hypothesis
            (for the first time) at each look time.
        ``'cumulative_power'`` : list of float
            The power of a sequential test to reject the null hypothesis by each
            look time. ``res['cumulative_power'][-1]`` is the power of the full
            sequential procedure.
        ``'uncorr_cumulative_power'`` : list of float
            Cumulative power if the rejection threshold at each look was not
            corrected using alpha-spending (as it should be).
        ``'n_expected'`` : float
            The expected sample size for the sequential procedure.
        ``'n_simulations'`` : int
            The number of bootstrap resamples used.
        ``'n_orig_data'`` : int
            The sample size of the original data ``X``, i.e. ``X.shape[0]``.
        ``'conditional'`` : bool
            Whether the power analysis that was run was conditional (``True``)
            or prospective (``False``).
        ``'test_func'`` : str
            Name of the sequential test function used.
        ``'test_func_kwargs'`` : dict
            A record of the arguments passed to the test function, including
            ``look_times`` and ``n_max``.

    Notes
    --------
    The significance level of the test used is specified in ``test_func``, and
    thus can be modified by passing an argument to ``test_func`` using
    ``**test_func_kwargs``.

    References
    ---------
    .. [1] Rubin, D. B. (1981). The bayesian bootstrap.
        The annals of statistics, 130-134.
    .. [2] Tu, D. & Zheng, Z. (1987). The Edgeworth expansion for the random
        weighting method. Chinese J. Appl. Probability and Statist., 3, 340-347.
    .. [3] Shao, J., & Tu, D. (2012). The jackknife and bootstrap.
        Springer Science & Business Media.
    .. [4] Ruzzoli, M., Torralba, M., Fernández, L. M., & Soto-Faraco, S. (2019).
        The relevance of alpha phase in human perception. Cortex, 120, 249-268.
    .. [5] Lakens, D., & Evers, E. R. (2014). Sailing from the seas of chaos
        into the corridor of stability: Practical recommendations to increase
        the informational value of studies.
        Perspectives on psychological science, 9(3), 278-292.





    '''
    def boot_stat(x, **kwargs): # thinly wrap test function
        _, ps, adj_alphas, _ = test_func(x, **kwargs)
        return np.stack([ps <= alpha, ps <= adj_alphas], axis = 1)
    test_func_kwargs['look_times'] = look_times
    test_func_kwargs['n_max'] = n_max
    test_func_kwargs['alpha'] = alpha
    if 'verbose' not in test_func_kwargs:
        test_func_kwargs['verbose'] = False
    rejections = _bootstrap(
        X, boot_stat,
        max(look_times), n_simulations,
        conditional, seed, n_jobs, **test_func_kwargs
    )

    # return simulation results with some metadata for posterity
    results = dict()
    results['uncorr_instantaneous_power'] = rejections[...,0].mean(0).tolist()
    results['rejection_probability'] = _prob_rejection(rejections[...,1]).tolist()
    results['cumulative_power'] = _cumulative_power(rejections[...,1]).tolist()
    results['uncorr_cumulative_power'] = _cumulative_power(rejections[...,0]).tolist()
    results['n_expected'] = _expected_sample_size(rejections[...,1], look_times)
    results['n_simulations'] = n_simulations
    results['n_orig_data'] = X.shape[0]
    results['conditional'] = False
    results['test_func'] = test_func.__name__
    test_func_kwargs = { # convert functions and classes to strings
        key: test_func_kwargs[key].__name__
            if (isfunction(test_func_kwargs[key]) or isclass(test_func_kwargs[key]))
            else test_func_kwargs[key]
            for key in test_func_kwargs
    }
    test_func_kwargs = { # convert SpendingFunction instances to strings
        key: test_func_kwargs[key].__class__.__name__
            if issubclass(type(test_func_kwargs[key]), SpendingFunction)
            else test_func_kwargs[key]
            for key in test_func_kwargs
    }
    results['test_func_kwargs'] = test_func_kwargs
    return results

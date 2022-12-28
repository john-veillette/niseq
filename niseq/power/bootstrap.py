from ..spending_functions import SpendingFunction
from mne.parallel import parallel_func
from mne.utils import check_random_state
from inspect import isclass, isfunction
import numpy as np

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
                conditional = False, seed = 0, n_jobs = 1, **kwargs):
    '''
    performs Bayesian bootstrap, optionally conditioned on current data
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

def bootstrap_predictive_power_1samp(X, test_func, look_times, n_max,
                conditional = False, n_simulations = 1024, seed = 0,
                n_jobs = 1, **test_func_kwargs):
    '''
    Computes the Bayesian predictive power non-parametrically using the Bayesian
    bootstrap. Optionally, you can condition on the current data to get
    conditional power, which is useful for adaptive designs.
    '''
    if 'y' in test_func_kwargs:
        raise Exception('Independent-sample test input, one-sample expected!')
    def boot_stat(x, **kwargs): # thinly wrap test function
        _, ps, adj_alphas, _ = test_func(x, **kwargs)
        return np.stack([ps <= .05, ps <= adj_alphas], axis = 1)
    test_func_kwargs['look_times'] = look_times
    test_func_kwargs['n_max'] = n_max
    rejections = _bootstrap(
        X, boot_stat,
        max(look_times), n_simulations,
        conditional, seed, n_jobs, **test_func_kwargs
    )

    # return simulation results with some metadata for posterity
    results = {}
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

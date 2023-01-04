from mne.utils import check_random_state
from mne.parallel import parallel_func
from ..max_test import (
    sequential_permutation_t_test_1samp,
    sequential_permutation_test_indep,
    sequential_permutation_test_corr
)
import numpy as np

N_SIMULATIONS = 500

def one_simulation(seed, tail = 0, sim_type = '1samp',
                   look_times = np.linspace(100, 300, 3).astype(int)):

    # generate null data
    rng = check_random_state(seed)
    x = rng.normal(loc = 0, size = look_times[-1])

    # run sequential test
    if sim_type == 'indep':
        conds = rng.choice([0, 1], look_times[-1])
        _, p, adj_alpha, _ = sequential_permutation_test_indep(
            x, conds, look_times, n_max = look_times[-1],
            tail = tail,
            seed = seed
        )
    elif sim_type == '1samp':
        _, p, adj_alpha, _ = sequential_permutation_t_test_1samp(
            x, look_times, n_max = look_times[-1],
            tail = tail,
            seed = seed
        )
    elif sim_type == 'corr':
        y = rng.rand(look_times[-1])
        _, p, adj_alpha, _ = sequential_permutation_test_corr(
            x, y, look_times, n_max = look_times[-1],
            tail = tail,
            seed = seed
        )

    # reject if p-val crosses sig threshold at any look time
    return np.array([np.any(p < .05), np.any(p < adj_alpha)])

def fpr_by_simulation(n_simulations, tail, sim_type, n_jobs = -1):
    parallel, p_func, _ = parallel_func(one_simulation, n_jobs)
    out = parallel(p_func(seed, tail, sim_type) for seed in range(n_simulations))
    rejections = np.stack(out)
    fpr = rejections.mean(0)
    return fpr[1] # false positive rate with sequential correction



def test_1samp_fpr(n_simulations = N_SIMULATIONS, alpha = .05, slack = .01):
    np.random.seed(0)
    for tail in [0, 1, -1]:
        fpr = fpr_by_simulation(n_simulations, tail, sim_type = '1samp')
        assert(fpr <= alpha + slack)
    return None

def test_indep_fpr(n_simulations = N_SIMULATIONS, alpha = .05, slack = .01):
    np.random.seed(0)
    for tail in [0, 1, -1]:
        fpr = fpr_by_simulation(N_SIMULATIONS, tail, sim_type = 'indep')
        assert(fpr <= alpha + slack)
    return None

def test_corr_fpr(n_simulations = N_SIMULATIONS, alpha = .05, slack = .01):
    np.random.seed(0)
    for tail in [0, 1, -1]:
        fpr = fpr_by_simulation(n_simulations, tail, sim_type = 'corr')
        assert(fpr <= alpha + slack)
    return None

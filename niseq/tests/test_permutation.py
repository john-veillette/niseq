from .._permutation import generate_permutation_dist, find_thresholds
from mne.utils import check_random_state
import numpy as np


def fpr_by_simulation(alpha, tail = 0,
    indep = False, n_simulations = 500, seed = 0):

    rng = check_random_state(seed)
    look_times = np.linspace(20, 100, 5).astype(int)
    ct = 0 # count number of rejections

    for i in range(n_simulations):

        # generate null data
        x = rng.normal(loc = 0, size = look_times[-1])
        y = rng.choice([0, 1], x.size)

        # and compute permutation distribution
        stat_1samp = lambda _x: np.mean(_x[0])
        stat_indep = lambda _x, _y: np.mean(_x[0]) - np.mean(_y[0])
        obs, H0 = generate_permutation_dist(
            x, y if indep else None,
            look_times = look_times,
            statistic = stat_indep if indep else stat_1samp,
            n_permutations = 1000,
            seed = seed
        )

        # compute p-values for each look time
        obs_stats = np.array([obs[n] for n in look_times])
        if tail == 0:
            p = (np.abs(H0) >= np.abs(obs_stats)).mean(0)
        elif tail == 1:
            p = (H0 >= obs_stats).mean()
        elif tail == -1:
            p = (H0 <= obs_stats).mean()

        # and find the adjusted significance thresholds
        # for each look
        spending, adj_alpha = find_thresholds(
            H0, look_times,
            look_times[-1],
            alpha = alpha,
            tail = tail
        )

        # reject if p-val crosses sig threshold at any look time
        ct += np.any(p < adj_alpha) # and keep count of rejections

    return ct / n_simulations # proportion of simulations where null rejected

def test_1samp_fpr(alpha = .05, slack = .01):
    for tail in [0, 1, -1]:
        fpr = fpr_by_simulation(alpha, tail = tail)
        assert(fpr <= alpha + slack)
    return None

def test_indep_fpr(alpha = .05, slack = .01):
    for tail in [0, 1, -1]:
        fpr = fpr_by_simulation(alpha, indep = True, tail = tail)
        assert(fpr <= alpha + slack)
    return None

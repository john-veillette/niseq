# -*- coding: utf-8 -*-

# Authors: John Veillette <johnv@uchicago.edu>
#
# License: BSD-3-Clause

import numpy as np

def _correlation_stat_fun(X, y):
    '''
    computes correlation coefficients in vectorized manner for a substantial
    speedup vs. using scipy's implementation applied across axis

    Inputs
    -------
    X: an (n, num_tests) np.ndarray
    y: an (n, 1) np.ndarray

    Returns
    -------
    r: a (num_tests,) array of pearson's correlation coefficients
    '''
    Xm = np.mean(X,axis = 0)[np.newaxis, :]
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis = 0)
    r_den = np.sqrt(np.sum((X - Xm)**2, axis = 0) * np.sum((y - ym)**2))
    r = r_num / r_den
    return r

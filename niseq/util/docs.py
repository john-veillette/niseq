# -*- coding: utf-8 -*-

# Authors: John Veillette <johnv@uchicago.edu>
#
# License: BSD-3-Clause

from mne.utils.docs import docdict as docdict_mne
from typing import Callable, Dict, List, Tuple
from collections import defaultdict
import sys

##############################################################################
# Define standard documentation entries

# make defaultdict so that incorrect MNE versions don't cause fatal error in
# code even if docstrings render incorrectly
docdict = defaultdict(str, {key: docdict_mne[key] for key in docdict_mne})

docdict['look_times'] = '''
look_times : list of int
    Sample sizes at which statistical test is applied to the data, *in order*.
    Not to exceed ``max_n``.
'''

docdict['n_max'] = '''
n_max : int
    Sample size at which data collection is completed, regardless of whether the
    null hypothesis has been rejected.
'''

docdict['alpha'] = '''
alpha : float, default: 0.05
    Desired false positive rate after all looks at the data (i.e. at ``n_max``)
'''

docdict['spending_func'] = '''
spending_func : instance of SpendingFunction, default: LinearSpendingFunction
    An initialized instance of one of ``SpendingFunction``'s subclasses. This
    defines a monotonically increasing function such that f(0) = 0 and
    f(n_max) = alpha, determining how Type I error is distributed over
    sequential analyses. See [2, 3] for details and provided spending functions
    in ``niseq.spending_functions`` module.
'''

docdict['X'] = '''
X : array, shape (n_observations[, p][, q][, r])
    The data to be analyzed. The first dimension of the array is the number of
    observations; remaining dimensions comprise the size of a single observation.
    Observations must appear in the order in which they were collected.
'''

docdict['X_clust'] = '''
X : array, shape (n_observations, p[, q][, r])
    The data to be analyzed. The first dimension of the array is the number of
    observations; remaining dimensions comprise the size of a single observation.
    Observations must appear in the order in which they were collected.
    Note: that the last dimension of ``X`` should correspond to the dimension
    represented in the adjacency parameter (e.g., spectral data should be
    provided as ``(observations, frequencies, channels/vertices))``.
'''

docdict['y'] = '''
y : array, shape (n_observations,)
    Value of dependent variable associated with each observation in ``X``.
'''

docdict['tail'] = '''
tail : -1 or 0 or 1, default: 0
    If tail is 1, the alternative hypothesis is that the
    mean of the data is greater than 0 (upper tailed test).  If tail is 0,
    the alternative hypothesis is that the mean of the data is different
    than 0 (two tailed test).  If tail is -1, the alternative hypothesis
    is that the mean of the data is less than 0 (lower tailed test).
'''

docdict['tail_corr'] = '''
tail : -1 or 0 or 1, default: 0
    If tail is 1, the alternative hypothesis is that the
    correlation is greater than 0 (upper tailed test).  If tail is 0,
    the alternative hypothesis is that the correlation is different
    than 0 (two tailed test).  If tail is -1, the alternative hypothesis
    is that the correlation is less than 0 (lower tailed test).
'''

docdict['n_permutations'] = '''
n_permutations : int, default: 1024
        Number of permutations.
'''

docdict['stat_fun_t'] = '''
stat_fun : callable() | None, default: None
    Function called to calculate the test statistic. Must accept 1D-array as
    input and return a 1D array. If ``None`` (the default), uses t statistic.
'''

docdict['stat_fun_F'] = '''
stat_fun : callable() | None, default: None
    Function called to calculate the test statistic. Must accept 1D-array as
    input and return a 1D array. If ``None`` (the default), uses t statistic
    or F statistic if more than two groups.
'''

docdict['stat_fun_corr'] = '''
stat_fun : callable() | None, default: None
    Function called to calculate the test statistic. Must accept 1D-array as
    input and return a 1D array. If ``None`` (the default), computes Pearson
    correlation.
'''

docdict['labels'] = '''
labels : array, shape (n_observations,)
    Condition label associated with each observation in ``X``.
'''

# overwrite MNE's documentation to avoid messing with bibtex for now
docdict['threshold'] = '''
threshold : float | dict | None, default: None
    The so-called "cluster forming threshold" in the form of a test statistic
    (note: this is not an alpha level / "p-value").
    If numeric, vertices with data values more extreme than ``threshold`` will
    be used to form clusters. If ``None``, threshold will be chosen
    automatically to correspond to a p-value of 0.05 for the given number of
    observations (only valid when using default statistic). If ``threshold`` is
    a :class:`dict` (with keys ``'start'`` and ``'step'``) then threshold-free
    cluster enhancement (TFCE) will be used (see TFCE example and [6]).
'''

docdict['alpha_spending_explanation'] = '''
Distributes Type I error over multiple, sequential analyses of the data (at
interim sample sizes specified in ``look_times`` never to exceed ``max_n``)
using a permutation-based adaptation of the alpha-spending procedure introduced
by Lan and DeMets [1]. This allows data collection to be terminated before
``max_n`` is reached if there is enough evidence to reject the null hypothesis
at an interim analysis, without inflating the false positive rate. This provides
a principled way to determine sample size and can result in substantial
efficiency gains over a fixed-sample design (i.e. can acheive the same
statistical power with a smaller expected sample size) [2, 3].
'''

docdict['references_clust'] = '''
.. [1] Gordon Lan, K. K., & DeMets, D. L. (1983).
    Discrete sequential boundaries for clinical trials.
    Biometrika, 70(3), 659-663.
.. [2] Lakens, D. (2014). Performing high-powered studies efficiently with
    sequential analyses. European Journal of Social Psychology, 44(7), 701-710.
.. [3] Lakens, D., Pahlke, F., & Wassmer, G. (2021).
    Group Sequential Designs: A Tutorial.
    https://doi.org/10.31234/osf.io/x4azm
.. [4] Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of
    EEG-and MEG-data. Journal of neuroscience methods, 164(1), 177-190.
.. [5] Jona Sassenhagen and Dejan Draschkow. Cluster-based permutation tests of
    meg/eeg data do not establish significance of effect latency or location.
    Psychophysiology, 56(6):e13335, 2019. doi:10.1111/psyp.13335.
.. [6] Stephen M. Smith and Thomas E. Nichols. Threshold-free cluster
    enhancement: addressing problems of smoothing, threshold dependence and
    localisation in cluster inference.
    NeuroImage, 44(1):83–98, 2009. doi:10.1016/j.neuroimage.2008.03.061.
'''

docdict['references_maxtype'] = '''
.. [1] Gordon Lan, K. K., & DeMets, D. L. (1983).
    Discrete sequential boundaries for clinical trials.
    Biometrika, 70(3), 659-663.
.. [2] Lakens, D. (2014). Performing high‐powered studies efficiently with
    sequential analyses. European Journal of Social Psychology, 44(7), 701-710.
.. [3] Lakens, D., Pahlke, F., & Wassmer, G. (2021).
    Group Sequential Designs: A Tutorial.
    https://doi.org/10.31234/osf.io/x4azm
.. [4] Thomas E. Nichols and Andrew P. Holmes. Nonparametric permutation tests
    for functional neuroimaging: a primer with examples.
    Human Brain Mapping, 15(1):1–25, 2002. doi:10.1002/hbm.1058.
'''

docdict['returns_clust'] = '''
looks : dict
    Dictionary containing results of each look at the data, indexed by the
    values provided in ``look_times``. Each entry of the dictionary is a tuple
    that contains:

    ``obs`` : array, shape (p[, q][, r])
        Statistic observed for all variables.
    ``clusters`` : list
        List type defined by out_type above.
    ``cluster_pv`` : array
        P-value for each cluster.
    ``H0`` : array, shape (n_permutations,)
        Max cluster level stats observed under permutation.

ps : array, shape (n_looks,)
    The lowest p-value obtained at each look specied in ``look_times``. These
    can be compared to ``adj_alphas`` to determine on which looks, if any, one
    can reject the null hypothesis.
adj_alphas: array, shape (n_looks,)
    The adjusted significance thresholds for each look, chosen to control the
    false positive rate across multiple, sequential analyses. All p-values
    should be compared to the adjusted alpha for the look at which they were
    computed.
spending: array, shape (n_looks,)
    The value of the alpha spending function at each look.
'''

docdict['returns_maxtype'] = '''
looks : dict
    Dictionary containing results of each look at the data, indexed by the
    values provided in ``look_times``. Each entry of the dictionary is a tuple
    that contains:

    ``obs`` : array of shape (p[, q][, r])
        Test statistic observed for all variables.
    ``p_values`` : array of shape (p[, q][, r])
        P-values for all the tests (a.k.a. variables).
    ``H0`` : array of shape [n_permutations]
        Max test statistics obtained by permutations.

ps : array, shape (n_looks,)
    The lowest p-value obtained at each look specied in ``look_times``. These
    can be compared to ``adj_alphas`` to determine on which looks, if any, one
    can reject the null hypothesis.
adj_alphas: array, shape (n_looks,)
    The adjusted significance thresholds for each look, chosen to control the
    false positive rate across multiple, sequential analyses. All p-values
    should be compared to the adjusted alpha for the look at which they were
    computed.
spending: array, shape (n_looks,)
    The value of the alpha spending function at each look.
'''

docdict['alpha_spending_note'] = '''
The number of and timing of looks at the data need not be planned in advance
(other than ``n_max``), but it is important to include all looks that have
already occured in ``look_times`` each time you analyze the data to ensure that
valid adjusted significance thresholds are computed. In your final analysis,
``look_times`` should contain the ordered sample sizes at all looks at the data
that occured during the study.

When reporting results, you should minimally include the sample sizes at each
look, the minimum p-values at each look, the adjusted significance thresholds
for each look (to which the p-values are compared), and the value of the
alpha-spending function at each look. See [3] for further recommendations.
'''

##############################################################################
# Below is copied from pycrostates/utils/_docs.py
# Defines functions to fill in placeholders in docstrings.

docdict_indented: Dict[int, Dict[str, str]] = {}

def fill_doc(f: Callable) -> Callable:
    """Fill a docstring with docdict entries.
    Parameters
    ----------
    f : callable
        The function to fill the docstring of (modified in place).
    Returns
    -------
    f : callable
        The function, potentially with an updated __doc__.
    """
    docstring = f.__doc__
    if not docstring:
        return f

    lines = docstring.splitlines()
    indent_count = _indentcount_lines(lines)

    try:
        indented = docdict_indented[indent_count]
    except KeyError:
        indent = " " * indent_count
        docdict_indented[indent_count] = indented = dict()

        for name, docstr in docdict.items():
            lines = [
                indent + line if k != 0 else line
                for k, line in enumerate(docstr.strip().splitlines())
            ]
            indented[name] = "\n".join(lines)

    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")

    return f


def _indentcount_lines(lines: List[str]) -> int:
    """Minimum indent for all lines in line list.
    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0
    """
    indent = sys.maxsize
    for k, line in enumerate(lines):
        if k == 0:
            continue
        line_stripped = line.lstrip()
        if line_stripped:
            indent = min(indent, len(line) - len(line_stripped))
    if indent == sys.maxsize:
        return 0
    return indent


def copy_doc(source: Callable) -> Callable:
    """Copy the docstring from another function (decorator).
    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.
    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.
    Parameters
    ----------
    source : callable
        The function to copy the docstring from.
    Returns
    -------
    wrapper : callable
        The decorated function.
    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B(A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         ''' this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """

    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise RuntimeError(
                f"The docstring from {source.__name__} could not be copied "
                "because it was empty."
            )
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func

    return wrapper

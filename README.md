[![Unit Tests](https://github.com/john-veillette/niseq/actions/workflows/pytest.yml/badge.svg)](https://github.com/john-veillette/niseq/actions/workflows/pytest.yml) [![codecov](https://codecov.io/gh/john-veillette/niseq/branch/main/graph/badge.svg?token=Q88ZWUEO7D)](https://codecov.io/gh/john-veillette/niseq) [![DOI](https://zenodo.org/badge/549239051.svg)](https://zenodo.org/badge/latestdoi/549239051) [![Downloads](https://static.pepy.tech/personalized-badge/niseq?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=PyPi%20downloads)](https://pypi.org/project/niseq/)
# niseq

`niseq` provides sequential generalizations of common statistical tests used in neuroimaging. That is, you can analyze your data multiple times throughout your experiment and stop data collection when there is enough evidence to reject the null hypothesis [without inflating your false positive rate](https://github.com/john-veillette/niseq/blob/b5c9c5f205aff35592a92628e1069f998f21a093/notebooks/FPR-simulation.ipynb) using an approach called [alpha spending](https://doi.org/10.1002/ejsp.2023).

The alpha spending approach to sequential analysis was first introduced by [Lan and DeMets (1983)](https://doi.org/10.1093/biomet/70.3.659) and has become common practice in clinical trials due to its substantial efficiency advantage over fixed-sample designs (i.e. fewer observations required on average to acheive the same statistical power). However, the original alpha spending approach relies on normality assumptions to derive adjusted significance thresholds, limiting its applicability to statistical tests used in neuroimaging. Our [permutation-based approach](https://github.com/john-veillette/niseq/blob/main/notebooks/permutation-explanation.ipynb) to alpha spending relaxes these assumptions, allowing essentially any fixed-sample permutation test (e.g. cluster-based permutation test, threshold-free cluster enhancement, network-based statistic, _t_-max and _F_-max) to be generalized to a sequentially-valid permutation test.

You may be interested in using `niseq` if you want to run a well-powered neuroimaging study and
* don't currently have a good way of estimating the sample size you need for your study a priori. __You can use a sequential stopping rule to determine your final sample size without inflating your false positive rate.__
* do have a good way of estimating a fixed sample size, but would prefer to collect fewer observations if justified by the data. __Sequential designs can acheive the same statistical power as a fixed-sample design using, on average, fewer observations.__
* want to use an adaptive design in case so you can adjust mid-experiment if you've underestimated the needed sample size. __You can conduct a conditional power analysis at an interim look at the data and adjust your design accordingly without inflating your false positive rate.__

## Installation

A stable version can be installed using
```
pip install niseq
```

and the development version using 
```
pip install git+https://github.com/john-veillette/niseq.git
```

## Usage

See our [API documentation](http://niseq.readthedocs.io/) and [example notebooks](https://github.com/john-veillette/niseq/tree/main/notebooks) for usage instructions.

Tutorial notebooks currently include:
* [Sequential cluster-based permutation tests and sequential threshold-free cluster enhancement on EEG data](https://github.com/john-veillette/niseq/blob/main/notebooks/EEG-cluster-and-TFCE-example.ipynb)
* [Sequential cluster-based permutation tests and sequential _t_-max on fMRI data](https://github.com/john-veillette/niseq/blob/main/notebooks/fMRI-cluster-and-tmax-example.ipynb)
* [Sequential network-based statistic for fMRI connectivity data](https://github.com/john-veillette/niseq/blob/main/notebooks/fMRI-NBS-example.ipynb)
* [Power analysis (both a priori & conditional) by bootstrap for fixed-sample, sequential, and adaptive designs](https://github.com/john-veillette/niseq/blob/main/notebooks/power-example.ipynb)

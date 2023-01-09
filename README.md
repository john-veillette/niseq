[![Unit Tests](https://github.com/john-veillette/niseq/actions/workflows/pytest.yml/badge.svg)](https://github.com/john-veillette/niseq/actions/workflows/pytest.yml) [![codecov](https://codecov.io/gh/john-veillette/niseq/branch/main/graph/badge.svg?token=Q88ZWUEO7D)](https://codecov.io/gh/john-veillette/niseq)
# niseq

`niseq` provides sequential generalizations of common statistical tests used in neuroimaging. That is, you can analyze your data multiple times throughout your experiment and stop data collection when there is enough evidence to reject the null hypothesis [without inflating your false positive rate](https://github.com/john-veillette/niseq/blob/b5c9c5f205aff35592a92628e1069f998f21a093/notebooks/FPR-simulation.ipynb) using an approach called [alpha spending](https://doi.org/10.1002/ejsp.2023). 

The alpha spending approach to sequential analysis was first introduced by [Lan and DeMets (1983)](https://doi.org/10.1093/biomet/70.3.659) and has become common practice in clinical trials. However, the original alpha spending approach relies on normality assumptions to derive adjusted significance thresholds, limiting its applicability to statistical tests used in neuroimaging. Our [permutation-based approach](https://github.com/john-veillette/niseq/blob/main/notebooks/permutation-explanation.ipynb) to alpha spending relaxes these assumptions, allowing essentially any fixed-sample permutation test (e.g. cluster-based permutation test, threshold-free cluster enhancement, network-based statistic, _t_-max and _F_-max) to be generalized to a sequentially-valid permutation test.

## Usage

See our [API documentation](http://niseq.readthedocs.io/) and [example notebooks](https://github.com/john-veillette/niseq/tree/main/notebooks) for usage instructions.

Example notebooks currently include:
* [Sequential cluster-based permutation tests and sequential threshold-free cluster enhancement on EEG data](https://github.com/john-veillette/niseq/blob/main/notebooks/EEG-cluster-and-TFCE-example.ipynb)
* [Sequential cluster-based permutation tests and sequential _t_-max on fMRI data](https://github.com/john-veillette/niseq/blob/main/notebooks/fMRI-cluster-and-tmax-example.ipynb)
* [Sequential network-based statistic for fMRI connectivity data](https://github.com/john-veillette/niseq/blob/main/notebooks/fMRI-NBS-example.ipynb)
* [Power analysis by bootstrap for fixed-sample and sequential designs](https://github.com/john-veillette/niseq/blob/main/notebooks/power-example.ipynb)

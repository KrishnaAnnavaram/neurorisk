import pytest

from neurorisk.config import Settings
from neurorisk.data import SyntheticSource, load
from neurorisk.features import load_feature_sets


@pytest.fixture(scope="session")
def dataset():
    return load(SyntheticSource(n_rows=400, seed=7))


@pytest.fixture(scope="session")
def feature_sets():
    return load_feature_sets()


@pytest.fixture()
def fast_settings():
    return Settings(seed=3, outer_folds=3, inner_folds=2, n_bootstrap=100, permutation_repeats=2)

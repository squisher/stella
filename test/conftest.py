import pytest


def pytest_addoption(parser):
    parser.addoption("--bench", action="store_true",
                     help="run benchmark tests")


def pytest_runtest_setup(item):
    if 'bench' in item.keywords and not item.config.getoption("--bench"):
        pytest.skip("need --bench option to run")

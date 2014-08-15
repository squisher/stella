import pytest


def pytest_addoption(parser):
    parser.addoption('-B', "--bench", action="store",
                     type=str, default=False,
                     help="run benchmark tests: short, or long")


def pytest_runtest_setup(item):
    if 'bench' in item.keywords and not item.config.getoption("--bench"):
        pytest.skip("need --bench option to run")
    bench = item.config.getoption("--bench")
    if bench not in (False, 'short', 'long', 's', 'l'):
        raise Exception("Invalid --bench option: " + bench)


def pytest_configure(config):
    pass

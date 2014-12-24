import pytest
from collections import defaultdict


def pytest_addoption(parser):
    parser.addoption('-B', "--bench", action="store",
                     type=str, default=False,
                     help="run benchmark tests: short, or long")


results = defaultdict(dict)


@pytest.fixture(scope="module")
def bench_result():
    return results

def pytest_runtest_setup(item):
    if 'bench' in item.keywords and not item.config.getoption("--bench"):
        pytest.skip("need --bench option to run")
    bench = item.config.getoption("--bench")
    if bench not in (False, 'short', 'long', 's', 'l'):
        raise Exception("Invalid --bench option: " + bench)


def pytest_configure(config):
    pass


def pytest_terminal_summary(terminalreporter):
    tr = terminalreporter
    lines = []
    name_width = max(map(len, results.keys())) + 2
    for benchmark, times in results.items():
        r = ['{}={:0.2f}'.format(i, t) for i, t in times.items()]
        lines.append("{}  {}".format(benchmark.ljust(name_width), ',  '.join(r)))
    if len(lines) > 0:
        tr.write_line('-'*len(lines[0]), yellow=True)
    for line in lines:
        tr.write_line(line)

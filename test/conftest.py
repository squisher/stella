import pytest
from collections import defaultdict


def pytest_addoption(parser):
    parser.addoption('-B', "--bench", action="store",
                     type=str, default=False,
                     help="run benchmark tests: veryshort, short, or long")
    parser.addoption('-E', "--extended-bench", action="store_true",
                     default=False,
                     help="run also extended benchmark tests: in Python, and with clang")


results = defaultdict(dict)


@pytest.fixture(scope="module")
def bench_result():
    return results


def pytest_runtest_setup(item):
    if 'bench' in item.keywords and not item.config.getoption("--bench"):
        pytest.skip("need --bench option to run")
    bench = item.config.getoption("--bench")
    if bench not in (False, 'short', 'long', 'veryshort', 's', 'l', 'v'):
        raise Exception("Invalid --bench option: " + bench)


def pytest_configure(config):
    pass


def pytest_terminal_summary(terminalreporter):
    tr = terminalreporter
    if not tr.config.getoption("--bench"):
        return
    lines = []
    name_width = max(map(len, results.keys())) + 2
    for benchmark, times in results.items():
        r = []
        s = []
        for impl, t in times.items():
            r.append('{}={:0.3f}s'.format(impl, t))
            if not impl.startswith('stella'):
                s.append('{}={:0.2f}x '.format('f'.rjust(len(impl)),
                                               t / times['stella']))
            else:
                s.append(' ' * len(r[-1]))

        lines.append("{}  {}".format(benchmark.ljust(name_width), ',  '.join(r)))
        lines.append("{}  {}".format(' '.ljust(name_width), '   '.join(s)))
    if len(lines) > 0:
        tr.write_line('-'*len(lines[0]), yellow=True)
    for line in lines:
        tr.write_line(line)

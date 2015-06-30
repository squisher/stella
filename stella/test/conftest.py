# Copyright 2013-2015 David Mohr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from collections import defaultdict


def pytest_addoption(parser):
    parser.addoption('-B', "--bench", action="store",
                     type=str, default=False,
                     help="run benchmark tests: veryshort, short, or long")
    parser.addoption('-E', "--extended-bench", action="count",
                     default=False,
                     help="run also extended benchmark tests: in Python, and with clang")


results = defaultdict(dict)


@pytest.fixture(scope="module")
def bench_result():
    return results


def pytest_runtest_setup(item):
    if 'bench' in item.keywords and not item.config.getoption("--bench"):
        pytest.skip("need --bench option to run")


def pytest_configure(config):
    bench = config.getoption("--bench")
    if bench not in (False, 'short', 'long', 'veryshort', 's', 'l', 'v'):
        raise Exception("Invalid --bench option: " + bench)


def save_results():
    import pickle
    with open('timings.pickle', 'wb') as f:
        pickle.dump(results, f)


def pytest_terminal_summary(terminalreporter):
    tr = terminalreporter
    if not tr.config.getoption("--bench"):
        return
    lines = []
    if results:
        name_width = max(map(len, results.keys())) + 2
        save_results()
    else:
        # TODO we were aborted, display a notice?
        name_width = 2
    for benchmark, type_times in sorted(results.items()):
        type_width = max(map(len, type_times.keys())) + 2
        for b_type, times in sorted(type_times.items()):
            r = []
            s = []
            for impl, t in times.items():
                r.append('{}={:0.3f}s'.format(impl, t))
                if not impl.startswith('stella'):
                    s.append('{}={:0.2f}x '.format('f'.rjust(len(impl)), t /
                                                   times['stella']))
                else:
                    s.append(' ' * len(r[-1]))

            lines.append("{} {}  {}".format(benchmark.ljust(name_width),
                                            b_type.ljust(type_width), ' '.join(r)))
            lines.append("{} {}  {}".format(' '.ljust(name_width),
                                            ' '.ljust(type_width), ' '.join(s)))

    if len(lines) > 0:
        tr.write_line('-'*len(lines[0]), yellow=True)
    for line in lines:
        tr.write_line(line)

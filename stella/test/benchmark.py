#!/usr/bin/env python
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

import os
import os.path
from subprocess import check_call, Popen, PIPE
import time

import pystache

from . import *  # noqa
import stella

opt = 3
min_speedup = 0.75


def ccompile(fn, src, cc=None, flags={}):
    """
    Write the string src into the file fn, then compile it with -O{opt} and
    return the executable name.
    """
    with open(fn, 'w') as f:
        f.write(src)

    if 'c' not in flags:
        flags['c'] = []
    if 'ld' not in flags:
        flags['ld'] = []

    if cc is None:
        if 'CC' in os.environ:
            CC = os.environ['CC']
        else:
            CC = 'gcc'
    else:
        CC = cc

    (root, ext) = os.path.splitext(fn)
    if os.path.exists(root):
        os.unlink(root)
    obj = root + ".o"
    if os.path.exists(obj):
        os.unlink(obj)
    with open(fn, 'rb') as f:
        sourcecode = f.read()

    # the following three cmds are equivalent to
    # [CC, '-Wall', '-O' + str(opt)] + flags + ['-o', root, fn]

    cmd = [CC] + flags['c'] + ['-Wall', '-E', '-o', '-', '-']
    print("Preprocessing: {0}".format(" ".join(cmd)))
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    preprocessed, serr = p.communicate(timeout=30, input=sourcecode)
    assert (not serr or not serr.decode())

    # start with C input, generate assembly
    cmd = [CC, '-Wall'] + flags['c'] + ['-x', 'cpp-output', '-S',
                                        '-O' + str(opt), '-o', '-', '-']
    print("Compiling to assembly: {0}".format(" ".join(cmd)))

    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    time_start = time.time()
    sout, serr = p.communicate(timeout=30, input=preprocessed)
    elapsed = time.time() - time_start

    assert not serr.decode()

    cmd = [CC] + flags['ld'] + ['-o', root, '-x', 'assembler', '-']
    print("Compiling to machine code & linking: {0}".format(" ".join(cmd)))
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    sout, serr = p.communicate(timeout=30, input=sout)
    assert (not serr or not serr.decode()) and (not sout or not sout.decode())

    return root, elapsed


def bench_it(name, c_src, args, extended=0, stella_f=None, full_f=None, flags={}):
    """args = {k=v, ...}
    Args gets expanded to `k`_init: `k`=`v` for the C template
    """
    if not stella_f and not full_f:
        raise Exception(
            "Either need to specify stella_f(*arg_value) or full_f(args, stats)")

    t_run = {}
    t_compile = {}

    c_args = {k+'_init': k+'='+str(v) for k, v in args.items()}
    print("Doing {0}({1})".format(name, args))
    src = pystache.render(c_src, **c_args)

    if extended:
        CCs = ['gcc', 'clang']
    else:
        CCs = ['gcc']

    for cc in CCs:
        exe, elapsed_compile = ccompile(__file__ + "." + name + ".c", src, cc, flags)
        t_compile[cc] = elapsed_compile

        cmd = [exe]

        print("Running C/{}: {}".format(cc, " ".join(cmd)))
        time_start = time.time()
        check_call(cmd)
        elapsed_c = time.time() - time_start
        t_run[cc] = elapsed_c

    print("Running Stella:")
    stats = {}
    wrapper_opts = {'debug': False, 'opt': opt, 'stats': stats}
    if stella_f:
        time_start = time.time()
        print(stella.wrap(stella_f, **wrapper_opts)
              (*[v for k, v in args.items()]))
        elapsed_stella = time.time() - time_start
    else:
        elapsed_stella = full_f(args, stella.wrap, wrapper_opts)

    t_run['stella'] = stats['elapsed']
    # TODO no need to keep track of the combined time, is there?
    # t_run['stella+compile'] = elapsed_stella
    t_compile['stella'] = elapsed_stella - stats['elapsed']

    if extended > 1:
        print("Running Python:")
        if stella_f:
            time_start = time.time()
            print(stella_f(*[v for k, v in args.items()]))
            elapsed_py = time.time() - time_start
        else:
            elapsed_py = full_f(args, time_stats, wrapper_opts)
        t_run['python'] = elapsed_py

    return {'run': t_run, 'compile': t_compile}


def bench_fib(duration, extended):
    from .langconstr import fib

    args = {'x': duration}
    fib_c = """
#include <stdio.h>
#include <stdlib.h>

long long fib(long long x) {
    if (x <= 2) {
        return 1;
    } else {
        return fib(x-1) + fib(x-2);
    }
}

int main(int argc, char ** argv) {
    long long r = 0;
    const int {{x_init}};

    r += fib(x);

    printf ("%lld\\n", r);
    exit (0);
}
"""

    return bench_it('fib', fib_c, args, extended, stella_f=fib)


def bench_fib_nonrecursive(duration, extended):
    from .langconstr import fib_nonrecursive

    args = {'x': duration}
    fib_c = """
#include <stdio.h>
#include <stdlib.h>

long long fib(long long x) {
    if (x == 0)
        return 1;
    if (x == 1)
        return 1;
    long long grandparent = 1;
    long long parent = 1;
    long long me = 0;
    int i;
    for (i=2; i<x; i++) {
        me = parent + grandparent;
        grandparent = parent;
        parent = me;
    }
    return me;
}

int main(int argc, char ** argv) {
    long long r = 0;
    const int {{x_init}};

    r += fib(x);

    printf ("%lld\\n", r);
    exit (0);
}
"""

    return bench_it('fib_nonrecursive', fib_c, args, extended, stella_f=fib_nonrecursive)


def bench_vs_template(module, extended, name, args, flags):
    fn = "{}/template.{}.{}.c".format(os.path.dirname(__file__),
                                      os.path.basename(__file__),
                                      name)
    with open(fn) as f:
        src = f.read()

    def run_it(args, wrapper, wrapper_opts):
        run_f, transfer, result_f = module.prepare(args)
        if transfer is None:
            transfer = []

        time_start = time.time()
        wrapper(run_f, **wrapper_opts)(*transfer)
        elapsed_stella = time.time() - time_start
        print(result_f(*transfer))

        return elapsed_stella

    return bench_it(name, src, args, extended, flags=flags, full_f=run_it)


def bench_si1l1s(module, extended, suffix, duration):
    args = {'seed': int(time.time() * 100) % (2**32),
            'rununtiltime': duration
            }
    return bench_vs_template(module, extended, 'si1l1s_' + suffix, args, {'ld': ['-lm']})


def bench_si1l1s_globals(duration, extended):
    from . import si1l1s_globals
    return bench_si1l1s(si1l1s_globals, extended, 'globals', duration)


def bench_si1l1s_struct(duration, extended):
    from . import si1l1s_struct
    return bench_si1l1s(si1l1s_struct, extended, 'struct', duration)


def bench_si1l1s_obj(duration, extended):
    from . import si1l1s_obj
    # reuse the 'struct' version of C since there is no native OO
    return bench_si1l1s(si1l1s_obj, extended, 'struct', duration)


def bench_nbody(n, extended):
    from . import nbody
    args = {'n': n,
            'dt': 0.01,
            }
    return bench_vs_template(nbody, extended, 'nbody', args, flags={'ld': ['-lm']})


def bench_heat(n, extended):
    from . import heat
    args = {'nsteps': n}
    return bench_vs_template(heat, extended, 'heat', args, flags={'ld': ['-lm'], 'c': ['-std=c99']})


def speedup(bench):
    return bench['run']['gcc'] / bench['run']['stella']


@bench
def test_fib(bench_result, bench_opt, bench_ext):
    duration = [30, 45, 48][bench_opt]
    bench_result['fib'] = bench_fib(duration, bench_ext)
    assert speedup(bench_result['fib']) >= min_speedup


@mark.skipif(True, reason="Runs too fast to be a useful benchmark")
def test_fib_nonrecursive(bench_result, bench_opt, bench_ext):
    duration = [50, 150, 175][bench_opt]
    bench_result['fib_nonrec'] = bench_fib_nonrecursive(duration, bench_ext)
    assert speedup(bench_result['fib_nonrec']) >= min_speedup


si1l1s_durations = ['1e5', '1e8', '1.2e9']


@bench
def test_si1l1s_globals(bench_result, bench_opt, bench_ext):
    duration = si1l1s_durations[bench_opt]
    bench_result['si1l1s_global'] = bench_si1l1s_globals(duration, bench_ext)
    assert speedup(bench_result['si1l1s_global']) >= min_speedup


@bench
def test_si1l1s_struct(bench_result, bench_opt, bench_ext):
    duration = si1l1s_durations[bench_opt]
    bench_result['si1l1s_struct'] = bench_si1l1s_struct(duration, bench_ext)
    assert speedup(bench_result['si1l1s_struct']) >= min_speedup


@bench
def test_si1l1s_obj(bench_result, bench_opt, bench_ext):
    duration = si1l1s_durations[bench_opt]
    bench_result['si1l1s_obj'] = bench_si1l1s_obj(duration, bench_ext)
    assert speedup(bench_result['si1l1s_obj']) >= min_speedup


@bench
def test_nbody(bench_result, bench_opt, bench_ext):
    duration = [250000, 10000000, 100000000][bench_opt]
    bench_result['nbody'] = bench_nbody(duration, bench_ext)
    assert speedup(bench_result['nbody']) >= min_speedup


@bench
def test_heat(bench_result, bench_opt, bench_ext):
    duration = [13, 3000, 50000][bench_opt]
    bench_result['heat'] = bench_heat(duration, bench_ext)
    assert speedup(bench_result['heat']) >= min_speedup

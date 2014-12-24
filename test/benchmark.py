#!/usr/bin/env python

import os
import os.path
from subprocess import call
import time

import pystache

from test import *  # noqa
import stella

opt = 3
min_speedup = 0.8


def ccompile(fn, src, cc=None, flags=[]):
    """
    Write the string src into the file fn, then compile it with -O{opt} and
    return the executable name.
    """
    with open(fn, 'w') as f:
        f.write(src)

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
    cmd = [CC, '-Wall', '-O' + str(opt)] + flags + ['-o', root, fn]
    print("Compiling: {0}".format(" ".join(cmd)))
    call(cmd)
    return root


def bench_it(name, c_src, args, extended=False, stella_f=None, full_f=None, flags=[]):
    """args = {k=v, ...}
    Args gets expanded to `k`_init: `k`=`v` for the C template
    """
    if not stella_f and not full_f:
        raise Exception(
            "Either need to specify stella_f(*arg_value) or full_f(args, stats)")

    r = {}

    c_args = {k+'_init': k+'='+str(v) for k, v in args.items()}
    print("Doing {0}({1})".format(name, args))
    src = pystache.render(c_src, **c_args)

    if extended:
        CCs = ['gcc', 'clang']
    else:
        CCs = ['gcc']

    for cc in CCs:
        exe = ccompile(__file__ + "." + name + ".c", src, cc, flags)

        cmd = [exe]
        print("Running C/{}: {}".format(cc, " ".join(cmd)))
        time_start = time.time()
        call(cmd)
        elapsed_c = time.time() - time_start
        r[cc] = elapsed_c

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

    r['stella'] = stats['elapsed']
    r['stella+compile'] = elapsed_stella

    if extended:
        print("Running Python:")
        if stella_f:
            time_start = time.time()
            print(stella_f(*[v for k, v in args.items()]))
            elapsed_py = time.time() - time_start
        else:
            elapsed_py = full_f(args, time_stats, wrapper_opts)
        r['python'] = elapsed_py

    return r


def bench_fib(duration, extended):
    from test.langconstr import fib

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

    return bench_it(name, src, args, extended, flags=['-lm'], full_f=run_it)


def bench_si1l1s(module, extended, suffix, duration):
    args = {'seed': '42',
            'rununtiltime': duration
            }
    return bench_vs_template(module, extended, 'si1l1s_' + suffix, args, ['-lm'])


def bench_si1l1s_globals(duration, extended):
    import test.si1l1s_globals
    return bench_si1l1s(test.si1l1s_globals, extended, 'globals', duration)


def bench_si1l1s_struct(duration, extended):
    import test.si1l1s_struct
    return bench_si1l1s(test.si1l1s_struct, extended, 'struct', duration)


def bench_si1l1s_obj(duration, extended):
    import test.si1l1s_obj
    # reuse the 'struct' version of C since there is no native OO
    return bench_si1l1s(test.si1l1s_obj, extended, 'struct', duration)


def bench_nbody(n, extended):
    import test.nbody
    args = {'n': n,
            'dt': 0.01,
            }
    return bench_vs_template(test.nbody, extended, 'nbody', args, flags=['-lm'])


def speedup(bench):
    return bench['stella'] / bench['gcc']


@bench
def test_fib(bench_result, bench_opt, bench_ext):
    duration = [30, 45, 48][bench_opt]
    bench_result['fib'] = bench_fib(duration, bench_ext)
    assert speedup(bench_result['fib']) >= min_speedup


si1l1s_durations = ['1e5', '1e6', '1e8']


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

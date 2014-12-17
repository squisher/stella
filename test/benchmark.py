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


def ccompile(fn, src, flags=[]):
    """
    Write the string src into the file fn, then compile it with -O{opt} and
    return the executable name.
    """
    with open(fn, 'w') as f:
        f.write(src)

    if 'CC' in os.environ:
        CC = os.environ['CC']
    else:
        CC = 'gcc'

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


def bench_it(name, c_src, args, stella_f=None, full_f=None, flags=[]):
    """args = {k=v, ...}
    Args gets expanded to `k`_init: `k`=`v` for the C template
    """
    if not stella_f and not full_f:
        raise Exception(
            "Either need to specify stella_f(*arg_value) or full_f(args, stats)")

    c_args = {k+'_init': k+'='+str(v) for k, v in args.items()}
    print("Doing {0}({1})".format(name, args))
    src = pystache.render(c_src, **c_args)
    exe = ccompile(__file__ + "." + name + ".c", src, flags)

    cmd = [exe]
    print("Running C:", " ".join(cmd))
    time_start = time.time()
    call(cmd)
    elapsed_c = time.time() - time_start

    print("Running Stella:")
    stats = {}
    if stella_f:
        time_start = time.time()
        print(stella.wrap(stella_f, debug=False, opt=opt, stats=stats)
              (*[v for k, v in args.items()]))
        elapsed_stella = time.time() - time_start
    else:
        elapsed_stella = full_f(args, stats)
    return (elapsed_c, stats['elapsed'], elapsed_stella)


def print_it(f, arg=None):
    print("Benchmarking {0} -O{1}".format(f.__name__, opt))
    (time_c, time_stella, time_stella_whole) = f(arg)
    speedup = time_c / time_stella
    print("Elapsed C: {0:2.2f}s\t Elapsed Stella: {1:2.2f}s\t Speed-Up: {2:2.2f}\t (Stella+Compile: {3:2.2f}s)".format(  # noqa
        time_c, time_stella, speedup, time_stella_whole))
    return speedup


def bench_fib(duration):
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

    return bench_it('fib', fib_c, args, stella_f=fib)


def bench_vs_template(module, name, args, flags):
    fn = "{}/template.{}.{}.c".format(os.path.dirname(__file__),
                                      os.path.basename(__file__),
                                      name)
    with open(fn) as f:
        src = f.read()

    def run_it(args, stats):
        run_f, transfer, result_f = module.prepare(args)
        if transfer is None:
            transfer = []

        time_start = time.time()
        stella.wrap(run_f, debug=False, opt=opt, stats=stats)(*transfer)
        elapsed_stella = time.time() - time_start
        print(result_f(*transfer))

        return elapsed_stella

    return bench_it(name, src, args, flags=['-lm'], full_f=run_it)


def bench_si1l1s(module, suffix, duration):
    args = {'seed': '42',
            'rununtiltime': duration
            }
    return bench_vs_template(module, 'si1l1s_' + suffix, args, ['-lm'])


def bench_si1l1s_globals(duration):
    import test.si1l1s_globals
    return bench_si1l1s(test.si1l1s_globals, 'globals', duration)


def bench_si1l1s_struct(duration):
    import test.si1l1s_struct
    return bench_si1l1s(test.si1l1s_struct, 'struct', duration)


def bench_si1l1s_obj(duration):
    import test.si1l1s_obj
    # reuse the 'struct' version of C since there is no native OO
    return bench_si1l1s(test.si1l1s_obj, 'struct', duration)


def bench_nbody(n):
    import test.nbody
    args = {'n': n,
            'dt': 0.01,
            }
    return bench_vs_template(test.nbody, 'nbody', args, flags=['-lm'])


@bench
def test_fib(bench_opt):
    duration = [45, 47][bench_opt]
    speedup = print_it(bench_fib, duration)
    assert speedup >= min_speedup


@bench
def test_si1l1s_globals(bench_opt):
    duration = ['1e6', '1e8'][bench_opt]
    speedup = print_it(bench_si1l1s_globals, duration)
    assert speedup >= min_speedup


@bench
def test_si1l1s_struct(bench_opt):
    duration = ['1e6', '1e8'][bench_opt]
    speedup = print_it(bench_si1l1s_struct, duration)
    assert speedup >= min_speedup


@bench
def test_si1l1s_obj(bench_opt):
    duration = ['1e6', '1e8'][bench_opt]
    speedup = print_it(bench_si1l1s_obj, duration)
    assert speedup >= min_speedup


@bench
def test_nbody(bench_opt):
    duration = [10000000, 50000000][bench_opt]
    speedup = print_it(bench_nbody, duration)
    assert speedup >= min_speedup

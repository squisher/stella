#!/usr/bin/env python

import os
import os.path
from subprocess import call
from time import time

import pystache

import stella

opt = 3


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
    """args = [(k,v),(k,v),...]"""
    if not stella_f and not full_f:
        raise Exception(
            "Either need to specify stella_f(*arg_value) or full_f(args, stats)")

    print("Doing {0}({1})".format(name, args))
    extra_args = []
    for k,v in args:
        if k.endswith('_init'):
            extra_args.append((k[:-5]+'_decl', v.split('=')[0]))
    src = pystache.render(c_src, **dict(args+extra_args))
    exe = ccompile(__file__ + "." + name + ".c", src, flags)

    cmd = ['./' + exe]
    print("Running C:", " ".join(cmd))
    time_start = time()
    call(cmd)
    elapsed_c = time() - time_start

    print("Running Stella:")
    stats = {}
    if stella_f:
        time_start = time()
        print(stella.wrap(stella_f, debug=False, opt=opt, stats=stats)
              (*[v for k, v in args]))
        elapsed_stella = time() - time_start
    else:
        elapsed_stella = full_f(args, stats)
    return (elapsed_c, stats['elapsed'], elapsed_stella)


def print_it(f):
    print("Benchmarking {0} -O{1}".format(f.__name__, opt))
    (time_c, time_stella, time_stella_whole) = f()
    print("Elapsed C: {0:2.2f}s\t Elapsed Stella: {1:2.2f}s\t Speed-Up: {2:2.2f}\t (Stella+Compile: {3:2.2f}s)".format(  # noqa
        time_c, time_stella, time_c / time_stella, time_stella_whole))


def bench_fib():
    from test.langconstr import fib_harness

    args = [('n', 4), ('x', 47)]
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
    int i;
    long long r = 0;

    for (i=0; i<{{n}}; i++) {
        r += fib({{x}});
    }

    printf ("%lld\\n", r);
    exit (0);
}
"""

    return bench_it('fib', fib_c, args, stella_f=fib_harness)


def bench_si1l1s(module, suffix):
    args = [('seed_init', 'seed=42'),
            ('rununtiltime_init', 'rununtiltime=1e6'),
            ]
    name = 'si1l1s_' + suffix
    fn = "{}/template.{}.{}.c".format(os.path.dirname(__file__),
                                      os.path.basename(__file__),
                                      name)
    with open(fn) as f:
        src = f.read()

    def run_si1l1s(args, stats):
        params = [v for k, v in args]
        s = module.Settings(params)
        module.prepare(s)

        time_start = time()
        stella.wrap(module.run, debug=False, opt=opt, stats=stats)()
        elapsed_stella = time() - time_start
        print(module.observations)

        return elapsed_stella

    return bench_it(name, src, args, flags=['-lm'], full_f=run_si1l1s)

def bench_si1l1s_globals():
    import test.si1l1s_globals
    return bench_si1l1s(test.si1l1s_globals, 'globals')


def bench_si1l1s_struct():
    import test.si1l1s_struct
    return bench_si1l1s(test.si1l1s_struct, 'struct')

if __name__ == '__main__':
    print_it(bench_fib)
    print('----------------------------------------')
    print_it(bench_si1l1s_globals)
    #print('----------------------------------------')
    #print_it(bench_si1l1s_struct)

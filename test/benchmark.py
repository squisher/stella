#!/usr/bin/env python

import os, os.path
from subprocess import call
import stella
from time import time

def ccompile(fn, src, opt=0):
    """
    Write the string src into the file fn, then compile it with -O{opt} and
    return the executable name.
    """
    with open(fn, 'w') as f:
        f.write(src)

    (root, ext) = os.path.splitext(fn)
    if os.path.exists(root):
        os.unlink(root)
    obj = root + ".o"
    if os.path.exists(obj):
        os.unlink(obj)
    cmd = ['clang','-Wall', '-O'+str(opt), '-o', root, fn]
    print ("Compiling: {0}".format(" ".join(cmd)))
    call(cmd)
    return root

def bench_fib(opt=0):
    from test.langconstr import fib_harness
    n = 4
    x = 45

    fib_c = """
#include <stdio.h>
#include <stdlib.h>

int fib(int x) {{
    if (x <= 2) {{
        return 1;
    }} else {{
        return fib(x-1) + fib(x-2);
    }}
}}

int main(int argc, char ** argv) {{
    int i;
    long long r = 0;

    for (i=0; i<{n}; i++) {{
        r += fib({x});
    }}

    printf ("%lld\\n", r);
    exit (0);
}}
""".format(n=n, x=x)

    print ("Doing {0} iterations of fib({1})".format(n,x))
    exe = ccompile(__file__+".fib.c", fib_c, opt)

    cmd = ['./'+exe]
    print ("Running C:", " ".join(cmd))
    time_start = time()
    call(cmd)
    elapsed_c = time() - time_start

    print ("Running Stella")
    stats = {}
    time_start = time()
    print(stella.wrap(fib_harness, debug=False, opt=opt, stats=stats)(n,x))
    elapsed_stella  = time() - time_start
    return (elapsed_c, stats['elapsed'], elapsed_stella)


if __name__ == '__main__':
    opt=2
    print ("Benchmarking -O{0}".format(opt))
    (time_c, time_stella, time_stella_whole) = bench_fib(opt)
    print("Elapsed C: {1:2.2f}s\t Elapsed Stella: {2:2.2f}s\t Speed-Up: {3:2.2f}\t (Stella+Compile: {4:2.2f}s)".format(opt, time_c, time_stella, time_c / time_stella, time_stella_whole))

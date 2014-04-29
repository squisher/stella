#!/usr/bin/env python

import stella.analysis as analysis
import stella.codegen as codegen

import logging
import faulthandler

_f = open('faulthandler.err', 'w')
faulthandler.enable(_f)
def zeros(shape=1, dtype=None):
    """Emulate certain features of `numpy.zeros`

    Note that `dtype` is ignored.
    """
    try:
        dim = len(shape)
        if dim == 1:
            shape=shape[0]
            raise TypeError()
    except TypeError:
        return [0 for i in range(shape)]

    # here dim > 1, build up the inner most dimension
    inner = [0 for i in range(shape[dim-1])]
    for i in range(dim-2,-1,-1):
        new_inner = [list(inner) for j in range(shape[i])]
        inner = new_inner
    return inner

def wrap(f, debug=True, ir=False, lazy=False, opt=None, stats=None):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    def run(*args):
        if stats == None:
            pass_stats = {}
        else:
            pass_stats = stats

        module = analysis.main(f, *args)
        prog = codegen.Program(module)

        prog.optimize(opt)

        if lazy:
            return prog
        elif ir:
            return str(prog.module)
        else:
            return prog.run(pass_stats)
    return run

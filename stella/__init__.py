#!/usr/bin/env python
import sys
import logging
import faulthandler

from . import analysis
from . import codegen

_f = open('faulthandler.err', 'w')
faulthandler.enable(_f)

def wrap(f, debug=True, ir=False, lazy=False, opt=None, stats=None):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    def run(*args, **kwargs):
        if stats == None:
            pass_stats = {}
        else:
            pass_stats = stats

        module = analysis.main(f, args, kwargs)
        prog = codegen.Program(module)

        prog.optimize(opt)

        if lazy:
            return prog
        elif ir:
            return prog.module.getLlvmIR()
        else:
            return prog.run(pass_stats)
    return run

# for convenience register the Python intrinsics directly in the stella namespace
# TODO maybe this isn't the best idea? It may be confusing. On the other hand,
# I don't plan to add more directly to the stella module.
from .intrinsics.python import *

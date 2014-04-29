#!/usr/bin/env python

import stella.analysis as analysis
import stella.codegen as codegen

import logging
import faulthandler

_f = open('faulthandler.err', 'w')
faulthandler.enable(_f)

def stella(f, debug=True, ir=False, lazy=False, opt=None, stats=None):
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

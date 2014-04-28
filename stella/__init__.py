#!/usr/bin/env python

import stella.analysis as analysis
import stella.codegen as codegen

import logging
import faulthandler

_f = open('faulthandler.err', 'w')
faulthandler.enable(_f)

logging.getLogger().setLevel(logging.DEBUG)

def stella(f, debug=False, opt=None, stats = None):
    def run(*args):
        if stats == None:
            pass_stats = {}
        else:
            pass_stats = stats

        module = analysis.main(f, *args)
        prog = codegen.Program(module)

        prog.optimize(opt)

        if not debug:
            return prog.run(pass_stats)
        elif debug == 'print':
            print(prog.module)
        else:
            return prog.module
    return run

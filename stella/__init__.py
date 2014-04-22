#!/usr/bin/env python

import stella.analysis as analysis
import stella.codegen as codegen

import logging
import faulthandler

_f = open('faulthandler.err', 'w')
faulthandler.enable(_f)

logging.getLogger().setLevel(logging.DEBUG)

def stella(f, debug=False):
    def run(*args):
        af = analysis.main(f, *args)
        prog = codegen.Program(af)
        if not debug:
            return prog.run()
        elif debug == 'print':
            print(prog.module)
        else:
            return prog.module
    return run

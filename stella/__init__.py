#!/usr/bin/env python

import stella.analysis as analysis
import stella.codegen as codegen

import logging
import faulthandler

faulthandler.enable()

logging.getLogger().setLevel(logging.DEBUG)

def stella(f):
    def run(*args):
        af = analysis.main(f, *args)
        return codegen.run(af, *args)
    return run

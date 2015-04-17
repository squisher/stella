#!/usr/bin/env python
import logging
import faulthandler

from . import analysis
from . import codegen
from . import utils

_f = open('faulthandler.err', 'w')
faulthandler.enable(_f)
logging.addLevelName(utils.VERBOSE, 'VERBOSE')


def wrap(f, debug=True, p=False, ir=False, lazy=False, opt=None, stats=None):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(utils.VERBOSE)

    def run(*args, **kwargs):
        if stats is None:
            pass_stats = {}
        else:
            pass_stats = stats

        module = analysis.main(f, args, kwargs)
        prog = codegen.Program(module)

        prog.optimize(opt)

        if lazy:
            return prog
        elif ir is True:
            return prog.getLlvmIR()
        elif type(ir) == str:
            print("Writing LLVM IR to {}...".format(ir))
            with open(ir, 'w') as fh:
                fh.write(prog.getLlvmIR())
            return
        elif p:
            print(prog.getLlvmIR())
        else:
            return prog.run(pass_stats)
    return run

# for convenience register the Python intrinsics directly in the stella
# namespace TODO maybe this isn't the best idea? It may be confusing. On the
# other hand, I don't plan to add more directly to the stella module.
# from .intrinsics.python import *

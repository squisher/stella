#!/usr/bin/env python
# Copyright 2013-2015 David Mohr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


def run_tests(args=None):
    import pytest
    import os.path
    try:
        from . import test
    except SystemError:
        from stella import test
    if args is None:
        args = os.path.dirname(test.__file__)
    pytest.main(args)

# for convenience register the Python intrinsics directly in the stella
# namespace TODO maybe this isn't the best idea? It may be confusing. On the
# other hand, I don't plan to add more directly to the stella module.
# from .intrinsics.python import *

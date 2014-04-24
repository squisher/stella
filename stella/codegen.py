#!/usr/bin/env python

import llvm
import llvm.core
import llvm.ee

import logging

from . import analysis
from .llvm import *
from .bytecode import *
from .exc import *

class Program(object):
    def __init__(self, af):
        self.af = af
        self.func = af.impl
        self.module = llvm.core.Module.new('__stella__')
        self.func.translate(self.module)
        self.createBlocks()

    def createBlocks(self):
        func = self.func.llvm
        # create blocks
        bb = func.append_basic_block("entry")

        for bc in self.af.bytecodes:
            if bc.discard:
                self.af.remove(bc)
                logging.debug("BLOCK skipped {0}".format(bc))
                continue

            newblock = ''
            if bc in self.af.incoming_jumps:
                assert not bc.block
                bc.block = func.append_basic_block(str(bc.loc))
                bb = bc.block
                newblock = ' NEW BLOCK (' + str(bc.loc) + ')'
            else:
                bc.block = bb
            logging.debug("BLOCK'D {0}{1}".format(bc, newblock))

        logging.debug("Printing all bytecodes:")
        self.af.bytecodes.printAll()

        logging.debug("Emitting code:")
        #import pdb; pdb.set_trace()
        bb = None
        for bc in self.af.bytecodes:
            if bb != bc.block:
                # new basic block, use a new builder
                builder = llvm.core.Builder.new(bc.block)

            bc.translate(self.module, builder)
            logging.debug("TRANS'D {0}".format(bc))
            # Note: the `and not' part is a basic form of dead code elimination
            #       This is used to drop unreachable "return None" which are implicitly added
            #       by Python to the end of functions.
            #       TODO is this the proper way to handle those returns? Any side effects?
            #            NEEDS REVIEW
            #       See also analysis.Function.analyze
            if isinstance(bc, BlockTerminal) and bc.next and bc.next not in self.af.incoming_jumps:
                logging.debug("TRANS stopping")
                #import pdb; pdb.set_trace()
                break

        self.llvm = self.makeStub()

    def makeStub(self):
        args = [llvm_constant(arg) for arg in self.func.arg_values]
        func_tp = llvm.core.Type.function(py_type_to_llvm(self.func.result.type), [])
        func = self.module.add_function(func_tp, self.af.getName()+'__stub__')
        bb = func.append_basic_block("entry")
        builder = llvm.core.Builder.new(bb)
        call = builder.call(self.func.llvm, args)
        builder.ret(call)
        return func

    def run(self):
        logging.debug("Verifying...")
        self.module.verify()

        logging.debug("Preparing execution...")

        #m = Module.new('-lm')
        #fntp = Type.function(Type.float(), [Type.int()])
        #func = m.add_function(fntp, '__powidf2')

        import ctypes
        from llvmpy import _api
        clib = ctypes.cdll.LoadLibrary(_api.__file__)
        logging.debug(str(clib))

        #import pdb; pdb.set_trace()

        # BUG: clib.__powidf2 gets turned into the following bytecode:
        # 103 LOAD_FAST                3 (clib) 
        # 106 LOAD_ATTR               11 (_Program__powidf2) 
        # 109 STORE_FAST               5 (f) 
        # which is not correct. I have no idea where _Program is coming from,
        # I'm assuming it is some internal Python magic going wrong
        f = getattr(clib,'__powidf2')

        logging.debug(str(f))

        llvm.ee.dylib_add_symbol('__powidf2', ctypes.cast(f, ctypes.c_void_p).value)

        #ee = ExecutionEngine.new(self.module)
        eb = llvm.ee.EngineBuilder.new(self.module)

        logging.debug("Enabling mcjit...")
        eb.mcjit(True)

        ee = eb.create()

        logging.debug("Arguments: {0}".format(list(zip(self.func.arg_types, self.func.arg_values))))

        # Now let's compile and run!
        logging.debug("Running...")
        retval = ee.run_function(self.llvm, [])

        logging.debug("Returning...")
        return llvm_to_py(self.func.result.type, retval)

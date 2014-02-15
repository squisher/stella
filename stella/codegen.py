#!/usr/bin/env python

from llvm import *
from llvm.core import *
from llvm.ee import *

import logging

from . import analysis
from .llvm import *
from .bytecode import *
from .exc import *

def example_jit(arg1_value, arg2_value):
    # Create a module, as in the previous example.
    my_module = Module.new('my_module')
    ty_int = Type.int()   # by default 32 bits
    ty_func = Type.function(ty_int, [ty_int, ty_int])
    f_sum = my_module.add_function(ty_func, "sum")
    f_sum.args[0].name = "a"
    f_sum.args[1].name = "b"
    bb = f_sum.append_basic_block("entry")
    builder = Builder.new(bb)
    tmp = builder.add(f_sum.args[0], f_sum.args[1], "tmp")
    builder.ret(tmp)

    # Create an execution engine object. This will create a JIT compiler
    # on platforms that support it, or an interpreter otherwise.
    ee = ExecutionEngine.new(my_module)

    # The arguments needs to be passed as "GenericValue" objects.
    arg1 = GenericValue.int(ty_int, arg1_value)
    arg2 = GenericValue.int(ty_int, arg2_value)

    # Now let's compile and run!
    retval = ee.run_function(f_sum, [arg1, arg2])

    # The return value is also GenericValue. Let's print it.
    #logging.debug("returned %d", retval.as_int())
    return retval.as_int()

class Program(object):
    def __init__(self, af, *args):
        self.af = af
        self.args = args
        self.module = Module.new('__stella__')
        self.arg_types = [py_type_to_llvm(arg.type) for arg in af.args]
        func_tp = Type.function(py_type_to_llvm(af.result.type), self.arg_types)
        self.func = self.module.add_function(func_tp, af.getName())

        for i in range(len(af.args)):
            self.func.args[i].name = af.args[i].name
            af.args[i].llvm = self.func.args[i]

        # create blocks
        bb = self.func.append_basic_block("entry")

        for bc in af.bytecodes:
            if bc.discard:
                af.remove(bc)
                logging.debug("BLOCK skipped {0}".format(bc))
                continue

            newblock = ''
            if bc.loc in af.incoming_jumps:
                assert not bc.block
                bc.block = self.func.append_basic_block(str(bc.loc))
                bb = bc.block
                newblock = ' NEW BLOCK (' + str(bc.loc) + ')'
            else:
                bc.block = bb
            logging.debug("BLOCK'D {0}{1}".format(bc, newblock))

        af.bytecodes.printAll()

        bb = None
        # emit code
        for bc in af.bytecodes:
            if bb != bc.block:
                # new basic block, use a new builder
                builder = Builder.new(bc.block)

            bc.translate(self.module, builder)
            logging.debug("TRANS'D {0}".format(bc))
            # Note: the `and not' part is a basic form of dead code elimination
            #       This is used to drop unreachable "return None" which are implicitly added
            #       by Python to the end of functions.
            #       TODO is this the proper way to handle those returns? Any side effects?
            #            NEEDS REVIEW
            #       See also analysis.Function.analyze
            if isinstance(bc, BlockTerminal) and bc.next and bc.next.loc not in af.incoming_jumps:
                logging.debug("TRANS stopping")
                #import pdb; pdb.set_trace()
                break

        self.func = self.makeStub()

    def makeStub(self):
        args = [llvm_constant(arg) for arg in self.args]
        func_tp = Type.function(py_type_to_llvm(self.af.result.type), [])
        func = self.module.add_function(func_tp, self.af.getName()+'__stub__')
        bb = func.append_basic_block("entry")
        builder = Builder.new(bb)
        call = builder.call(self.func, args)
        builder.ret(call)
        return func

    def run(self):
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

        dylib_add_symbol('__powidf2', ctypes.cast(f, ctypes.c_void_p).value)

        #ee = ExecutionEngine.new(self.module)
        eb = EngineBuilder.new(self.module)

        logging.debug("Enabling mcjit...")
        eb.mcjit(True)

        ee = eb.create()

        logging.debug("Arguments: {0}".format(list(zip(self.arg_types, self.args))))

        # Now let's compile and run!
        logging.debug("Running...")
        retval = ee.run_function(self.func, [])

        # The return value is also GenericValue. Let's print it.
        logging.debug("Returning...")
        return llvm_to_py(self.af.result.type, retval)

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
        func_tp = Type.function(py_type_to_llvm(af.return_tp), self.arg_types)
        self.func = self.module.add_function(func_tp, af.getName())

        for i in range(len(af.args)):
            self.func.args[i].name = af.args[i].name
            af.args[i].llvm = self.func.args[i]

        bb = self.func.append_basic_block("entry")
        builder = Builder.new(bb)

        for i, bc in af.labels.items():
            bc.block = self.func.append_basic_block(str(i))

        logging.debug("PyStack bytecode:")
        for bc in af.bytecodes:
            logging.debug(str(bc))
            if bc.block:
                builder = Builder.new(bc.block)
            if hasattr(bc, 'translate'):
                bc.translate(self.module, builder)
            else:
                # TODO Fix this exception
                raise UnimplementedException("Bytedcode {0} at {1} does not yet have a LLVM translation".format(bc))

    def run(self):
        logging.debug("Preparing execution...")
        ee = ExecutionEngine.new(self.module)

        logging.debug("Arguments: {0}".format(list(zip(self.arg_types, self.args))))
        # The arguments needs to be passed as "GenericValue" objects.
        llvm_args = [get_generic_value(tp, arg) for tp, arg in zip(self.arg_types, self.args)]

        # Now let's compile and run!
        logging.debug("Running...")
        retval = ee.run_function(self.func, llvm_args)

        # The return value is also GenericValue. Let's print it.
        logging.debug("Returning...")
        return llvm_to_py(self.af.return_tp, retval)

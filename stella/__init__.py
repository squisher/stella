#!/usr/bin/env python

from llvm import *
from llvm.core import *
from llvm.ee import *

import logging

import inspect

def example_jit(fn):
    def jit(arg1_value, arg2_value):
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

    return jit


# TODO
# actually call llvm!!
def stella(f):
    args = inspect.getargspec(f)
    return example_jit(f)
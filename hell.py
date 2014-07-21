#!/usr/bin/env python

from llvm import *
from llvm.core import *
from llvm.ee import *

import logging

def spawn():
    my_module = Module.new('my_module')
    tp_int = Type.int()   # by default 32 bits
    tp_func = Type.function(tp_int, [])
    tp_struct = Type.struct([tp_int, tp_int], name='test_struct')
    f_sum = my_module.add_function(tp_func, "sum")
    bb = f_sum.append_basic_block("entry")
    builder = Builder.new(bb)
    tmp = builder.add(Constant.int(tp_int, 42),
                      Constant.int(tp_int, 0),
                      "tmp")
    struct = builder.alloca(tp_struct)
    p = builder.gep(struct, [Constant.int(tp_int, 0), Constant.int(tp_int, 0)])
    builder.store(p, tmp)
    builder.ret(tmp)

    # Create an execution engine object. This will create a JIT compiler
    # on platforms that support it, or an interpreter otherwise.
    eb = EngineBuilder.new(my_module)
    eb.mcjit(True)
    ee = eb.create()

    # Now let's compile and run!
    retval = ee.run_function(f_sum, [])

    # The return value is also GenericValue. Let's print it.
    #logging.debug("returned %d", retval.as_int())
    return retval.as_int()

if __name__ == '__main__':
    print(spawn())

#!/usr/bin/env python

from llvm import *
from llvm.core import *
from llvm.ee import *

import logging

def playground():
    my_module = Module.new('my_module')
    tp_int = Type.int(64)   # By default 32bit, indices MUST be 64bit
    tp_func = Type.function(tp_int, [])
    f_sum = my_module.add_function(tp_func, "sum")
    bb = f_sum.append_basic_block("entry")
    builder = Builder.new(bb)
    tmp = builder.add(Constant.int(tp_int, 42),
                      Constant.int(tp_int, 0),
                      "tmp")


    # Q: How to access members of a freshly stack allocated struct?
    tp_struct = Type.struct([tp_int, tp_int], name='test_struct')
    struct = builder.alloca(tp_struct)
    print (str(struct) + '; ' + repr(struct))
    p = builder.gep(struct, [Constant.int(tp_int, 0), Constant.int(tp_int, 0)])
    builder.store(tmp, p)
    tmp2 = builder.load(p)


    print(str(my_module))
    builder.ret(tmp2)

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
    print(playground())

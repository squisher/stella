#!/usr/bin/env python

from llvm import *
from llvm.core import *
from llvm.ee import *

import logging
import ctypes
import cffi

def playground():
    my_module = Module.new('my_module')
    tp_int = Type.int(64)
    tp_idx = Type.int()
    tp_struct = Type.struct([tp_int, tp_int, tp_int, tp_int], name='test_struct')
    #tp_func = Type.function(Type.pointer(tp_struct), [])
    tp_func = Type.function(tp_int, [])
    f_sum = my_module.add_function(tp_func, "sum")
    bb = f_sum.append_basic_block("entry")
    builder = Builder.new(bb)
    tmp = builder.add(Constant.int(tp_int, 42),
                      Constant.int(tp_int, 0),
                      "tmp")


    #struct = builder.alloca(tp_struct)
    struct = builder.malloc(tp_struct)
    p = builder.gep(struct, [Constant.int(tp_idx, 0), Constant.int(tp_idx, 0)])
    builder.store(tmp, p)
    p = builder.gep(struct, [Constant.int(tp_idx, 0), Constant.int(tp_idx, 1)])
    builder.store(Constant.int(tp_int, 43), p)
    p = builder.gep(struct, [Constant.int(tp_idx, 0), Constant.int(tp_idx, 2)])
    builder.store(Constant.int(tp_int, 44), p)
    p = builder.gep(struct, [Constant.int(tp_idx, 0), Constant.int(tp_idx, 3)])
    builder.store(Constant.int(tp_int, 45), p)
    tmp2 = builder.ptrtoint(struct, tp_int)

    #p = builder.gep(struct, [Constant.int(tp_idx, 0), Constant.int(tp_idx, 0)])
    #tmp3 = builder.load(p)

    #builder.ret(struct)
    builder.ret(tmp2)
    #builder.ret(tmp3)

    print(str(my_module))
    eb = EngineBuilder.new(my_module)
    eb.mcjit(True)
    ee = eb.create()

    retval = ee.run_function(f_sum, [])

    #return retval.as_pointer()
    return retval.as_int()

class Point(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_int64),
        ('y', ctypes.c_int64),
        ('z', ctypes.c_int64),
        ('r', ctypes.c_int64),
    ]

if __name__ == '__main__':
    p = playground()
    print("Returned: " + str(p))
    #cast_p = ctypes.cast(p, ctypes.POINTER(ctypes.c_long))

    ffi = cffi.FFI()
    ffi.cdef("""typedef struct {
        int64_t x;
        int64_t y;
        int64_t z;
        int64_t r;
    } sFoo;""")
    x = ffi.cast("sFoo *", p)
    print("cffi:", x, x.x, x.y, x.z, x.r)

    cast_p = ctypes.cast(p, ctypes.POINTER(Point))
    #import pdb; pdb.set_trace()
    print("ctypes:", cast_p[0], cast_p[0].x, cast_p[0].y, cast_p[0].z, cast_p[0].r)

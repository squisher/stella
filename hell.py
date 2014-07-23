#!/usr/bin/env python

from llvm import *
from llvm.core import *
from llvm.ee import *

import logging
import ctypes
import cffi

def playground(point):
    my_module = Module.new('my_module')
    tp_int = Type.int(64)
    tp_idx = Type.int()
    tp_struct = Type.struct([tp_int, tp_int, tp_int, tp_int], name='test_struct')
    #tp_func = Type.function(Type.pointer(tp_struct), [])
    tp_func = Type.function(tp_int, [])
    f_sum = my_module.add_function(tp_func, "sum")
    bb = f_sum.append_basic_block("entry")
    builder = Builder.new(bb)

    addr = ctypes.addressof(point)
    addr_llvm = Constant.int(tp_int, int(addr))
    struct = builder.inttoptr(addr_llvm, Type.pointer(tp_struct))
    #print(str(struct))

    ione = Constant.int(tp_idx, 1)
    izero = Constant.int(tp_idx, 0)
    one = Constant.int(tp_int, 1)

    p = builder.gep(struct, [izero, izero])
    tmp = builder.load(p)
    res = builder.add(tmp, one)
    builder.store(res, p)

    p = builder.gep(struct, [izero, ione])
    tmp = builder.load(p)
    res = builder.add(tmp, one)
    builder.store(res, p)

    #p = builder.gep(struct, [Constant.int(tp_idx, 0), Constant.int(tp_idx, 0)])
    #tmp3 = builder.load(p)

    #builder.ret(struct)
    builder.ret(res)
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
    point = Point(1,2,3,4)
    p = playground(point)
    print("Returned: " + str(p))
    print(point, point.x, point.y, point.z, point.r)

    #cast_p = ctypes.cast(p, ctypes.POINTER(Point))
    #print("ctypes:", cast_p[0], cast_p[0].x, cast_p[0].y, cast_p[0].z, cast_p[0].r)

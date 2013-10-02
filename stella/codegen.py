#!/usr/bin/env python

from llvm import *
from llvm.core import *
from llvm.ee import *

from stella import analysis
from stella.bytecode import *

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

tp_int = Type.int()
tp_float = Type.float()
def py_type_to_llvm(tp):
    if tp == int:
        return tp_int
    elif tp == float:
        return tp_float
    else:
        raise TypeError("Unknown type " + tp)

def get_generic_value(tp, val):
    if type(val) == int:
        return GenericValue.int(tp, val)
    elif type(val) == float:
        return GenericValue.real(tp, val)

def llvm_to_py(tp, val):
    if tp == int:
        return val.as_int()
    elif tp == float:
        return val.as_real(py_type_to_llvm(tp), val)
    else:
        raise Exception ("Unknown type {0}".format(tp))

def run(af, *args):
    my_module = Module.new('__stella__')
    arg_types = [py_type_to_llvm(arg.type) for arg in af.args]
    func_tp = Type.function(py_type_to_llvm(af.return_tp), arg_types)
    my_func = my_module.add_function(func_tp, af.getName())

    for i in range(len(af.args)):
        my_func.args[i].name = af.args[i].name
        af.args[i].llvm = my_func.args[i]

    bb = my_func.append_basic_block("entry")
    builder = Builder.new(bb)
    #import pdb; pdb.set_trace()
    for bc in af.bytecodes:
        bc.translate(builder)

    ee = ExecutionEngine.new(my_module)

    # The arguments needs to be passed as "GenericValue" objects.
    #arg1 = GenericValue.int(ty_int, arg1_value)
    llvm_args = [get_generic_value(tp, arg) for tp, arg in zip(arg_types, args)]

    # Now let's compile and run!
    retval = ee.run_function(my_func, llvm_args)

    # The return value is also GenericValue. Let's print it.
    #logging.debug("returned %d", retval.as_int())
    return llvm_to_py(af.return_tp, retval)

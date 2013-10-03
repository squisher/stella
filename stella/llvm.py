from llvm import *
from llvm.core import *
from llvm.ee import *

import logging

tp_int = Type.int(64)
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
        return val.as_int_signed()
    elif tp == float:
        return val.as_real(py_type_to_llvm(tp))
    else:
        raise Exception ("Unknown type {0}".format(tp))



from llvm import *
from llvm.core import *
from llvm.ee import *

import logging

from .exc import *

tp_int = Type.int(64)
#tp_float = Type.float() # Python always works with double precision
tp_double = Type.double()

def py_type_to_llvm(tp):
    if tp == int:
        return tp_int
    elif tp == float:
        return tp_double
    else:
        raise TypingError("Unknown type " + tp)

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
        raise TypingError("Unknown type {0}".format(tp))

def llvm_constant(val):
    tp = type(val)
    if tp == int:
        return Constant.int(tp_int, val)
    elif tp == float:
        return Constant.real(tp_double, val)
    else:
        raise UnimplementedError("Unknown constant type {0}".format(tp))

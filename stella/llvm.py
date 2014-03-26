from llvm import *
from llvm.core import *
from llvm.ee import *

import logging

from .exc import *

tp_int = Type.int(64)
tp_int32 = Type.int(32)
#tp_float = Type.float() # Python always works with double precision
tp_double = Type.double()
tp_bool = Type.int(1)

def py_type_to_llvm(tp):
    """Map from Python types to LLVM types."""
    if tp == int:
        return tp_int
    elif tp == float:
        return tp_double
    elif tp == bool:
        return tp_bool
    else:
        raise TypingError("Unknown type " + str(tp))

def get_generic_value(tp, val):
    if type(val) == int:
        return GenericValue.int_signed(tp, val)
    elif type(val) == float:
        return GenericValue.real(tp, val)
    elif type(val) == bool:
        return GenericValue.int(tp, val)

def llvm_to_py(tp, val):
    if tp == int:
        return val.as_int_signed()
    elif tp == float:
        return val.as_real(py_type_to_llvm(tp))
    elif tp == bool:
        return bool(val.as_int())
    else:
        raise TypingError("Unknown type {0}".format(tp))

def llvm_constant(val):
    tp = type(val)
    if tp == int:
        return Constant.int(tp_int, val)
    elif tp == float:
        return Constant.real(tp_double, val)
    elif tp == bool:
        return Constant.int(tp_bool, val)
    # HACK {
    elif tp == None.__class__:
        return Constant.int(tp_int, 0)
    # } HACK
    else:
        raise UnimplementedError("Unknown constant type {0}".format(tp))

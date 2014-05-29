import llvm
import llvm.core
import llvm.ee

import logging

from .exc import *

class NoType(object):
    __name__ = '?'
    @classmethod
    def __str__(klass):
        return '<?>'
    # llvm.core.ArrayType does something funny, it will compare against _ptr,
    # so let's just add the attribute here to enable equality tests
    _ptr = None

class ArrayType(object):
    __name__ = 'ArrayType'
    tp = NoType
    shape = None
    def __init__(self, tp, shape):
        assert isinstance(tp, type)
        self.tp = tp
        self.shape = shape
    def make(self):
        return tp_array(self.tp, self.shape)
    def __str__(self):
        return "<{0}*{1}>".format(self.tp, self.shape)

tp_int = llvm.core.Type.int(64)
tp_int32 = llvm.core.Type.int(32)
#tp_float = llvm.core.Type.float() # Python always works with double precision
tp_double = llvm.core.Type.double()
tp_bool = llvm.core.Type.int(1)
tp_void = llvm.core.Type.void()
def tp_array(tp, n):
    return llvm.core.Type.array(py_type_to_llvm(tp), n)

py_types = [int, float, bool]
def supported_py_type(tp):
    if type(tp) == str:
        types = map(lambda x: x.__name__, py_types)
    else:
        types = py_types
    return any([tp == t for t in types])

def py_type_to_llvm(tp):
    """Map from Python types to LLVM types."""
    if tp == int:
        return tp_int
    elif tp == float:
        return tp_double
    elif tp == bool:
        return tp_bool
    elif tp == type(None):
        return tp_void
    elif type(tp) == ArrayType:
        # TODO is a pointer always required?
        return llvm.core.Type.pointer(tp.make())
    elif tp == NoType:
        raise TypingError("Insufficient knowledge to derive a type")
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
    elif tp == type(None):
        return None
    else:
        raise TypingError("Unknown type {0}".format(tp))

def llvm_constant(val):
    tp = type(val)
    if tp == int:
        return llvm.core.Constant.int(tp_int, val)
    elif tp == float:
        return llvm.core.Constant.real(tp_double, val)
    elif tp == bool:
        return llvm.core.Constant.int(tp_bool, val)
    elif tp == str:
        return llvm.core.Constant.string(val)
    # HACK {
    elif tp == None.__class__:
        return llvm.core.Constant.int(tp_int, 0)
    # } HACK
    else:
        raise UnimplementedError("Unknown constant type {0}".format(tp))
